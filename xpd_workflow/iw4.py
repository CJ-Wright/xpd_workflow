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


def process_to_iq(hdr):
    # 1. get a detector calibration
    geo = get_from_analysisstore(hdr,
                                 function_name='run_calibration',
                                 run_function=calibrate_detector,
                                 data_names='poni',
                                 uid_name='detector_calibration_uid')
    imgs = analysis_run_engine(hdr, subs_dark, 'subs_dark')
    # 3. polarization correction
    corrected_imgs = analysis_run_engine([imgs, geo], polarization_correction,
                                         'polarization_correction')
    # 4. mask
    masks = analysis_run_engine([corrected_imgs, geo], mask_img, 'mask_img')
    # 5. integrate
    iqs = analysis_run_engine([imgs, masks, geo], integrate, 'integrate')
    return iqs


def process_to_pdf(hdr, bg_hdr_idx=-1):
    iqs = process_to_iq()
    bg_hdrs = db(is_background=True, background_uid=hdr['background_uid'])
    bg_hdr = bg_hdrs[bg_hdr_idx]
    bg_iq = find_an_hdr(bg_hdr['uid'])
    if not bg_iq:
        process_to_iq(bg_hdr)

    # 6a. associate background
    associated_bg_hdr = analysis_run_engine([hdr, iqs, bg_hdr, bg_iq],
                                            associate_background,
                                            'associate_background')
    # 7. subtract background
    corrected_iqs = analysis_run_engine(associated_bg_hdr,
                                        background_subtraction,
                                        'background_subtraction')
    # 8., 9. optimize PDF params, get PDF
    pdf = analysis_run_engine(corrected_iqs,
                              optimize_pdf_parameters,
                              'optimize_pdf_parameters')
    # 10. Profit
    return pdf


def find_an_hdr(uid, function_name):
    # Search analysisstore for data which is descended from the uid and who's
    # analysis was produced by the function
    while True:
        for uid in uids:
            uids = conn.find_analysis_header()
    pass


def analysis_run_engine(hdrs, run_function, function_name, md=None, **kwargs):
    """
    Properly run an analysis function on a group of headers while recording
    the data into analysisstore

    Parameters
    ----------
    hdrs: list of MDS headers or a MDS header
        The headers or header to be analyzed, note that the headers are each
        used in the analysis, if you wish to run multiple headers through a
        pipeline you must
    run_function
    function_name
    md
    kwargs

    Returns
    -------

    """
    # write analysisstore header
    analysis_hdr_uid = conn.insert_analysis_header(
        uid=str(uuid4()),
        time=time.time(),
        provenance={'function_name': function_name,
                    'hdr_uids': [hdr['uid'] for hdr in hdrs]},
        **md)

    data_hdr = None
    exit_status = 'failure'
    # run the analysis function
    try:
        rf = run_function(*hdrs, **kwargs)
        for i, res, data_names, data_keys in enumerate(rf):
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
        exit_status = 'success'
    except:
        # Analysis failed!
        exit_status = 'failure'
    finally:
        conn.insert_analysis_tail(analysis_header=analysis_hdr_uid,
                                  uid=str(uuid4()),
                                  time=time.time(), exit_status=exit_status)
        return analysis_hdr_uid


def sample_analysis_function(hdr, **kwargs):
    # extract data from analysisstore or databroker
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
    data_keys = {'img': dict(
        source='subs_dark',
        external='FILESTORE:',
        dtype='array'
    )}

    data_names = ['img']
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
        yield uid, data_names, data_keys


def mask_img(hdr, cal_hdr,
             alpha=2.5, lower_thresh=0.0, upper_thresh=None,
             margin=30., tmsk=None):
    data_keys = {'msk': dict(
        source='auto_mask',
        external='FILESTORE:',
        dtype='array'
    )}

    data_names = ['msk']
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
        yield uid, data_names, data_keys


def polarization_correction(hdr, cal_hdr, polarization=.99):
    data_keys = {'img': dict(
        source='pyFAI-polarization',
        external='FILESTORE:',
        dtype='array'
    )}

    data_names = ['img']
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
        yield uid, data_names, data_keys


def integrate(img_hdr, mask_hdr, cal_hdr, stats='mean', npt=1500):
    if not isinstance(stats, list):
        stats = [stats]
    for stat in stats:
        data_keys = {'iq_{}'.format(stat): dict(
            source='cjw-integrate',
            external='FILESTORE:',
            dtype='array'
        )}
        data_names = ['iq_{}'.format(stat)]
    geo = next(get_analysis_events(cal_hdr, fill=True))['data']['poni']
    for img_event, mask_event in zip(get_analysis_events(img_hdr, fill=True),
                                     get_analysis_events(mask_hdr, fill=True)):
        mask = mask_event['data']['msk']
        img = img_event['data']['img'][mask]
        q = geo.qArray(img.shape)[mask] / 10  # pyFAI works in nm^1
        uids = []
        for stat in stats:
            iq, q, _ = sts.binned_statistic(q, img, statistic=stat, bins=npt)

            # save
            save_output(q, iq, 'save_loc', 'Q')

            # insert into filestore
            uid = str(uuid4())
            fs_res = fsc.insert_resource('CHI', 'file_loc')
            fsc.insert_datum(fs_res, uid)
            uids.append(uid)
        yield uids, data_names, data_keys


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
        yield uid, data_names, data_keys



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
#
# def subtract_dark(hdr, dark_hdr, img_name='pe1_img', dark_event_idx=-1):
#     dark_events = get_events(dark_hdr, fill=True)
#     dark_img = islice(dark_events, dark_event_idx)
#     events = get_events(hdr, fill=True)
#     imgs = [event['data'][img_name] - dark_img for event in events]
#     return imgs
#
#
# def calibrate_energy():
#     pass
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
