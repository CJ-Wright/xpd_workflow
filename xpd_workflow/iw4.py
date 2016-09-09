from databroker import db, get_events, get_table
from analysisstore.client.commands import AnalysisClient
import filestore.commands as fsc
from itertools import islice
from uuid import uuid4
import time
import sys
import subprocess as sbp
from skbeam.io.save_powder_output import save_output
import numpy as np
from tifffile.tifffile import imsave
import scipy.stats as sts
from skbeam.io.save_powder_output import save_output
import pyFAI

conf = dict(host='xf28id-ca1.cs.nsls2.local', port=7767)
conn = AnalysisClient(conf)

uid_is_pair_dict = {'detector_calibration_uid': 'is_detector_calibration',
                    'dark_hdr_uid': 'is_dark_img',
                    }


def get_analysis_events(hdr, fill=False):
    descriptors = conn.find_data_reference_header(analysis_header=hdr['uid'])

    events = conn.find_data_reference(data_reference_header=descriptors['uid'])

    for event in events:
        if fill:
            updated_event = event.copy()
            data = updated_event['data']
            for key in data.keys():
                if 'external' in descriptors[key] and descriptors[key][
                    'external'] == 'FILESTORE':
                    data[key] = fsc.retrieve(data[key])
        yield event


def calibrate_detector(hdr, nrg_cal_hdr_idx=-1):
    # 1. get energy calibration
    nrg_hdrs = db(is_energy_calibration=True,
                  energy_calibration_uid=hdr['energy_calibration_uid'])
    nrg_hdr = nrg_hdrs[nrg_cal_hdr_idx]
    nrg_cal_hdr = find_an_hdr(nrg_hdr['uid'], 'calibrate_energy')
    if not nrg_cal_hdr:
        nrg_cal_hdr = calibrate_energy(nrg_hdr)
    # 2. run detector calibration
    calibration = analysis_run_engine([hdr, nrg_cal_hdr], detector_calibrate)
    pass


def spoof_detector_calibration(cal_hdr, poni_file):
    calibration = analysis_run_engine([cal_hdr], spoof_det_cal,
                                      poni_file=poni_file)


def spoof_det_cal(hdr, poni_file):
    data_names = ['poni']
    data_keys = {k: dict(
        source='pyFAI-calib spoof',
        external='FILESTORE:',
        dtype='dict'
    ) for k in data_names}

    for event in get_analysis_events(hdr):
        uid = str(uuid4())
        fs_res = fsc.insert_resource('poni', 'poni_file')
        fsc.insert_datum(fs_res, uid)
        yield uid, data_names, data_keys, pyFAI.load(poni_file)


def process_to_iq(hdrs, det_cal_hdr_idx=-1):
    """
    Process raw data from MDS to I(Q) data

    Parameters
    ----------
    hdrs: header or list of headers
        The data to be processed
    det_cal_hdr_idx: int, optional
        Calibration index to use if there are multiple calibrations
        Defaults to the latest calibration

    Yields
    -------
    iqs: analysis header
        The headers associated with the I(Q) data
    """
    if not isinstance(hdrs, list):
        hdrs = [hdrs]
    for hdr in hdrs:
        # 1. get a detector calibration
        cal_hdrs = db(is_detector_calibration=True,
                      detector_calibration_uid=hdr['detector_calibration_uid'])
        cal_hdr = cal_hdrs[det_cal_hdr_idx]
        cal_geo_hdr = find_an_hdr(cal_hdr['uid'], 'calibrate_detector')
        if not cal_geo_hdr:
            cal_geo_hdr = calibrate_detector(cal_hdr)
        # 2. dark subtraction
        imgs = analysis_run_engine(hdr, subs_dark)
        # 3. polarization correction
        corrected_imgs = analysis_run_engine([imgs, cal_geo_hdr],
                                             polarization_correction)
        # 4. mask
        masks = analysis_run_engine([corrected_imgs, cal_geo_hdr], mask_img)
        # 5. integrate
        iqs = analysis_run_engine([imgs, masks, cal_geo_hdr], integrate)
        yield iqs


def process_to_pdf(hdrs, bg_hdr_idx=-1, det_cal_hdr_idx=-1):
    """
    Process a raw MDS header to the PDF

    Parameters
    ----------
    hdrs: header or list of headers
        The data to be processed
    bg_hdr_idx: int, optional
        Background index to use if there are multiple background headers
    det_cal_hdr_idx: int, optional
        Calibration index to use if there are multiple calibrations
        Defaults to the latest calibration

    Yields
    -------
    pdf: analysis header
        The headers associated with the PDF data
    """
    if not isinstance(hdrs, list):
        hdrs = [hdrs]
    for hdr in hdrs:
        iqs = process_to_iq(hdr, det_cal_hdr_idx=det_cal_hdr_idx)
        bg_hdrs = db(is_background=True, background_uid=hdr['background_uid'])
        bg_hdr = bg_hdrs[bg_hdr_idx]
        bg_iq_hdr = find_an_hdr(bg_hdr['uid'], 'integrate')
        if not bg_iq_hdr:
            bg_iq_hdr = process_to_iq(bg_hdr)

        # 6a. associate background
        associated_bg_hdr = analysis_run_engine([hdr, iqs, bg_hdr, bg_iq_hdr],
                                                associate_background)
        # 7. subtract background
        corrected_iqs = analysis_run_engine(associated_bg_hdr,
                                            background_subtraction)
        # 8., 9. optimize PDF params, get PDF
        pdf = analysis_run_engine(corrected_iqs, optimize_pdf_parameters)
        # 10. Profit
        yield pdf


def find_an_hdr(uid, function_name):
    # Search analysisstore for data which is descended from the uid and who's
    # analysis was produced by the function
    while True:
        for uid in uids:
            uids = conn.find_analysis_header()
    return hdrs


def analysis_run_engine(hdrs, run_function, md=None, subscription=None,
                        **kwargs):
    """
    Properly run an analysis function on a group of headers while recording
    the data into analysisstore

    Parameters
    ----------
    hdrs: list of MDS headers or a MDS header
        The headers or header to be analyzed, note that the headers are each
        used in the analysis, if you wish to run multiple headers through a
        pipeline you must
    run_function: generator
        This generator processes each event in the headers, it returns the
         filestore resource uids as a list of strings, the data names as a
         list of strings and the data keys as a dict
    md: dict
        Metadata to be added to the analysis header
    subscription: function or list of functions
        Run after processing the event, eg graphing output data
    kwargs: dict
        Additional arguments passed directly to the run_function

    Returns
    -------

    """
    # write analysisstore header
    analysis_hdr_uid = conn.insert_analysis_header(
        uid=str(uuid4()),
        time=time.time(),
        provenance={'function_name': run_function.__name__,
                    'hdr_uids': [hdr['uid'] for hdr in hdrs]},
        **md)

    data_hdr = None
    exit_md = {'exit_status': 'failure'}
    # run the analysis function
    try:
        rf = run_function(*hdrs, **kwargs)
        for i, res, data_names, data_keys, data in enumerate(rf):
            if not data_hdr:
                data_hdr = dict(analysis_header=analysis_hdr_uid,
                                data_keys=data_keys,
                                time=time.time(), uid=str(uuid4()))
                data_hdr_uid = conn.insert_data_reference_header(**data_hdr)
            conn.insert_data_reference(
                data_reference_header=data_hdr_uid,
                uid=str(uuid4()),
                time=time.time(),
                data={k: v for k, v in zip(data_names, res)},
                timestamps={},
                seq_num=i)
            if not isinstance(subscription, list):
                subscription = [subscription]
            for subs in subscription:
                subs(data)
        exit_md['exit_status'] = 'success'
    except Exception as e:
        # Analysis failed!
        exit_md['exit_status'] = 'failure'
        exit_md['exception'] = e
    finally:
        conn.insert_analysis_tail(analysis_header=analysis_hdr_uid,
                                  uid=str(uuid4()),
                                  time=time.time(), **exit_md)
        return analysis_hdr_uid


def sample_analysis_function(hdr, **kwargs):
    # extract data from databroker
    data_keys = {'sum': dict(
        source='pyFAI-calib',
        external='FILESTORE:',
        dtype='dict'
    )}
    data_names = ['sum', '1', '2']
    for event in get_events(hdr):
        res = [1 + 1, 'foo', 'bar']
        # save
        # insert into filestore
        yield res, data_names, data_keys


def subs_dark(hdr, dark_hdr_idx=-1, dark_event_idx=-1):
    data_names = ['img']
    data_keys = {k: dict(
        source='subs_dark',
        external='FILESTORE:',
        dtype='array'
    ) for k in data_names}

    dark_hdr = db(is_dark_img=True, dark_uid=hdr['dark_uid'])[dark_hdr_idx]
    dark_events = get_events(dark_hdr, fill=True)
    dark_img = islice(dark_events, dark_event_idx)
    for event in get_analysis_events(hdr, fill=True):
        light_img = event['data']['img']
        img = light_img - dark_img
        # save
        # Eventually save this as a sparse array to save space
        imsave('file_loc', img)

        # insert into filestore
        uid = str(uuid4())
        fs_res = fsc.insert_resource('TIFF', 'file_loc')
        fsc.insert_datum(fs_res, uid)
        yield uid, data_names, data_keys, img


def mask_img(hdr, cal_hdr,
             alpha=2.5, lower_thresh=0.0, upper_thresh=None,
             margin=30., tmsk=None):
    data_names = ['msk']
    data_keys = {k: dict(
        source='auto_mask',
        external='FILESTORE:',
        dtype='array'
    ) for k in data_names}

    geo = next(get_analysis_events(cal_hdr, fill=True))['data']['poni']
    for event in get_analysis_events(hdr, fill=True):
        img = event['data']['img']
        r = geo.rArray(img.shape)
        pixel_size = [getattr(geo, a) for a in ['pixel1', 'pixel2']]
        rres = np.hypot(*pixel_size)
        rbins = np.arange(np.min(r) - rres / 2., np.max(r) + rres / 2., rres)
        if tmsk is None:
            tmsk = np.ones(img.shape, dtype=int).astype(bool)
        if margin:
            tmsk *= margin_mask(img.shape, margin)
        if lower_thresh:
            tmsk *= (img > lower_thresh).astype(bool)
        if upper_thresh:
            tmsk *= (img < upper_thresh).astype(bool)
        if alpha:
            tmsk *= ring_blur_mask(img, r, alpha, rbins, mask=tmsk)
        # save
        # Eventually save this as a sparse array to save space
        np.save('file_loc', tmsk)

        # insert into filestore
        uid = str(uuid4())
        fs_res = fsc.insert_resource('npy', 'file_loc')
        fsc.insert_datum(fs_res, uid)
        yield uid, data_names, data_keys, tmsk


def polarization_correction(hdr, cal_hdr, polarization=.99):
    data_names = ['img']
    data_keys = {k: dict(
        source='pyFAI-polarization',
        external='FILESTORE:',
        dtype='array'
    ) in data_names}

    geo = next(get_analysis_events(cal_hdr, fill=True))['data']['poni']
    for event in get_analysis_events(hdr, fill=True):
        img = event['data']['img']
        img /= geo.polarization(img.shape, polarization)
        # save
        imsave('file_loc', img)

        # insert into filestore
        uid = str(uuid4())
        fs_res = fsc.insert_resource('TIFF', 'file_loc')
        fsc.insert_datum(fs_res, uid)
        yield uid, data_names, data_keys, img


def integrate(img_hdr, mask_hdr, cal_hdr, stats='mean', npt=1500):
    if not isinstance(stats, list):
        stats = [stats]
    data_names = ['iq_{}'.format(stat) for stat in stats]

    data_keys = {dn: dict(
        source='cjw-integrate',
        external='FILESTORE:',
        dtype='array'
    ) for dn in data_names}

    geo = next(get_analysis_events(cal_hdr, fill=True))['data']['poni']
    for img_event, mask_event in zip(get_analysis_events(img_hdr, fill=True),
                                     get_analysis_events(mask_hdr, fill=True)):
        mask = mask_event['data']['msk']
        img = img_event['data']['img'][mask]
        q = geo.qArray(img.shape)[mask] / 10  # pyFAI works in nm^1
        uids = []
        data = []
        for stat in stats:
            iq, q, _ = sts.binned_statistic(q, img, statistic=stat, bins=npt)
            data.append((q, iq))
            # save
            save_output(q, iq, 'save_loc', 'Q')

            # insert into filestore
            uid = str(uuid4())
            fs_res = fsc.insert_resource('CHI', 'file_loc')
            fsc.insert_datum(fs_res, uid)
            uids.append(uid)
        yield uids, data_names, data_keys, data


def associate_background(hdr, iqs, bg_hdr, bg_iq, match_key=None):
    data_names = ['foreground_iq', 'background_iq']
    data_keys = {k: dict(
        source='associate_background',
        external='FILESTORE:',
        dtype='array'
    ) for k in data_names}

    # mux the background data with the foreground data
    # TODO: get more complex, handle multiple match keys with a cost function
    bg_iq_events = get_analysis_events(bg_iq)
    if match_key is not None:
        table = get_table(bg_hdr, fields=match_key)
        for event, iq in zip(get_events(hdr), get_analysis_events(iqs)):
            bg_event_idx = np.argmin(
                np.abs(event[match_key] - table[match_key]))
            bg_event = next(islice(bg_iq_events, bg_event_idx))
            # TODO: support multiple iqs
            yield [iq['data']['iq_mean'], bg_event['data']['iq_mean']], \
                  data_names, data_keys


def background_subtraction(hdr, bg_scale=1):
    data_names = ['iq']
    data_keys = {k: dict(
        source='background_subtraction',
        external='FILESTORE:',
        dtype='array'
    ) for k in data_names}

    ran_manual = False

    for event in get_analysis_events(hdr, fill=True):
        fg_iq, bg_iq = [event['data'][k] for k in
                        ['foreground_iq', 'background_iq']]
        if bg_scale == 'auto':
            raise NotImplementedError(
                'There is no automatic background scaling,yet')
        if not ran_manual and bg_scale == 'manual':
            raise NotImplementedError('WE have not implemented slider bar'
                                      'based manual background scaling, yet')
        corrected_iq = fg_iq[1] - bg_iq[1] * bg_scale
        # save
        save_output(fg_iq[0], corrected_iq, 'file_loc', 'Q')

        # insert into filestore
        uid = str(uuid4())
        fs_res = fsc.insert_resource('CHI', 'file_loc')
        fsc.insert_datum(fs_res, uid)
        yield uid, data_names, data_keys, corrected_iq

# def calibrate_detector(hdr, **kwargs):
#     data_keys = {'poni': dict(
#         source='pyFAI-calib',
#         external='FILESTORE:',
#         dtype='dict'
#     )}
#     data_names = ['poni']
#     for event in get_events(hdr):
#         sbp.run(['pyFAI-calib', '-D {} -{} {} -c {} {}'.format(
#             detector, w_or_e, w_or_e_val, calibration_file, image_file
#         )])
#     energy = get_from_analysisstore(hdr,
#                                     function_name='calibrate_energy',
#                                     run_function=calibrate_energy,
#                                     uid_name='energy_calibration_uid',
#                                     data_names='energy')
#     # find subtracted dark data
#     img = get_from_analysisstore(hdr,
#                                  function_name='subtract_dark',
#                                  run_function=subtract_dark,
#                                  uid_name='dark_img_uid',
#                                  data_names='img',
#                                  fs_override=True)
#

# def get_from_analysisstore(hdr,
#                            function_name,
#                            run_function,
#                            uid_name=None,
#                            data_names=None,
#                            find_hdr_idx=-1,
#                            analysis_hdr_idx=-1, ev_idx=-1,
#                            override_analyssisstore=False,
#                            fs_override=False,
#                            **kwargs):
#     """
#     Retrieve data from analysisstore, if it doesn't exist run and insert the
#     data.
#
#     Parameters
#     ----------
#     hdr: mds.header
#         The MDS header for the data to be processed
#     function_name: str
#         Name of the function to be found or run
#     run_function: function
#         The function to be run if no data is found
#     uid_name: str, optional
#         The name of the uid to look up if we need to find data.
#         If none given only the hdr data used
#     data_names: list of str or str
#         List of names for the data coming out of analysisstore or being
#         inserted
#     find_hdr_idx: int, optional
#         The index of header to use if multiples are valid for the search
#         criteria.
#         Only needed if secondary data is needed for the analysis.
#         Defaults to the latest data
#     analysis_hdr_idx: int, optional
#         The index of analysis header to use if multiple are valid for the data
#         set.
#         Defaults to the most recent analysis
#     ev_idx: int, optional
#         The index of analysis header to use if multiple are valid for the data
#         set.
#         Defaults to the most recent analysis
#     override_analyssisstore
#     kwargs
#
#     Returns
#     -------
#     results:
#         The results of either the analysis function or the analysisstore
#         retrieval
#     """
#     # If we are running on data in the hdr
#     if not uid_name:
#     # Or we are putting two datasets together and need to find the second one
#     else:
#         uid = hdr[uid_name]
#         search_dict = {uid_name: uid, uid_is_pair_dict[uid_name]: True}
#         find_hdrs = db(**search_dict)
#         if not find_hdrs:
#             raise IndexError('This secondary data is not associated')
#         else:
#             find_hdr = find_hdrs[find_hdr_idx]
#         find_hdr_uid = find_hdr['uid']
#     analysis_hdrs = conn.find_analysis_header(run_header_uid=find_hdr_uid,
#                                               function=function_name)
#     # If it is not in analysisstore, run it
#     if not analysis_hdrs:
#         raise NotImplementedError
#     # Else withdraw it from the db
#     else:
#         a_hdr = analysis_hdrs[analysis_hdr_idx]
#         dref_hdr = conn.find_data_reference_header(analysis_header=a_hdr)
#         as_tail_hdr = conn.find_analysis_tail(analysis_header=a_hdr)
#         # If the prior run was bad re-run it
#         if as_tail_hdr['exit_staus'] != 'success':
#             if not analysis_hdrs:
#                 raise NotImplementedError
#         # Else retrieve it
#         else:
#             analysis_evs = conn.find_data_reference(
#                 data_reference_header=dref_hdr)
#             if not isinstance(data_names, list):
#                 data_names = [data_names]
#             results = [fsc.retrieve(analysis_evs[ev_idx]['data'][data_name])
#                        for data_name in data_names]
#             if len(results) == 1:
#                 results = results[0]
#             return results
