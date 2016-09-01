from databroker import db, get_events
from analysisstore.client.commands import AnalysisClient
import filestore.commands as fsc
from itertools import islice
from uuid import uuid4
import time
import sys

conf = dict(host='xf28id-ca1.cs.nsls2.local', port=7767)
conn = AnalysisClient(conf)

uid_is_pair_dict = {'detector_calibration_uid': 'is_detector_calibration',
                    'dark_hdr_uid': 'is_dark_img',
                    }


def process_to_pdf(hdr):
    # 1. get a detector calibration
    geo = get_from_analysisstore(hdr, function_name='run_calibration',
                                 run_function=calibrate_detector,
                                 data_names='poni',
                                 uid_name='detector_calibration_uid')
    # 2. subtract dark image
    imgs = get_from_analysisstore(hdr, run_dark_subtraction, 'img', 'dark_uid')
    # 3. polarization correction
    corrected_imgs = get_from_analysisstore(hdr, run_polarization_correction,
                                            'img', 'dark_uid', geo=geo)
    # 4. mask
    masks = get_from_analysisstore(hdr, run_automask, 'msk', 'dark_uid',
                                   alpha=3, margin=30)
    # 5. integrate
    iqs = get_from_analysisstore(hdr, run_dark_subtraction, 'iq', 'dark_uid',
                                 npt)
    # 6. get background
    bgd_iqs = get_background_data(hdr)
    # 7. subtract background
    corrected_iqs = get_background_corrected_iqs(hdr)
    # 8., 9. optimize PDF params, get PDF
    pdf = get_optimized_pdf_params(hdr)
    # 10. Profit
    pass


def get_from_analysisstore(hdr,
                           function_name,
                           run_function,
                           uid_name=None,
                           data_names=None,
                           find_hdr_idx=-1,
                           analysis_hdr_idx=-1, ev_idx=-1,
                           override_analyssisstore=False,
                           fs_override=False,
                           **kwargs):
    """
    Retrieve data from analysisstore, if it doesn't exist run and insert the
    data.

    Parameters
    ----------
    hdr: mds.header
        The MDS header for the data to be processed
    function_name: str
        Name of the function to be found or run
    run_function: function
        The function to be run if no data is found
    uid_name: str, optional
        The name of the uid to look up if we need to find data.
        If none given only the hdr data used
    data_names: list of str or str
        List of names for the data coming out of analysisstore or being
        inserted
    find_hdr_idx: int, optional
        The index of header to use if multiples are valid for the search
        criteria.
        Only needed if secondary data is needed for the analysis.
        Defaults to the latest data
    analysis_hdr_idx: int, optional
        The index of analysis header to use if multiple are valid for the data
        set.
        Defaults to the most recent analysis
    ev_idx: int, optional
        The index of analysis header to use if multiple are valid for the data
        set.
        Defaults to the most recent analysis
    override_analyssisstore
    kwargs

    Returns
    -------
    results:
        The results of either the analysis function or the analysisstore
        retrieval
    """
    # If we are running on data in the hdr
    if not uid_name:
        find_hdr_uid = hdr['uid']
        find_hdr = None
    # Or we are putting two datasets together and need to find the second one
    else:
        uid = hdr[uid_name]
        search_dict = {uid_name: uid, uid_is_pair_dict[uid_name]: True}
        find_hdrs = db(**search_dict)
        if not find_hdrs:
            raise IndexError('This secondary data is not associated')
        else:
            find_hdr = find_hdrs[find_hdr_idx]
        find_hdr_uid = find_hdr['uid']
    analysis_hdrs = conn.find_analysis_header(run_header_uid=find_hdr_uid,
                                              function=function_name)
    # If it is not in analysisstore, run it
    if not analysis_hdrs or override_analyssisstore:
        results = run_and_insert(hdr, find_hdr, function_name, run_function,
                                 **kwargs)
        return results
    # Else withdraw it from the db
    else:
        a_hdr = analysis_hdrs[analysis_hdr_idx]
        dref_hdr = conn.find_data_reference_header(analysis_header=a_hdr)
        as_tail_hdr = conn.find_analysis_tail(analysis_header=a_hdr)
        # If the prior run was bad re-run it
        if as_tail_hdr['exit_staus'] != 'success':
            results = run_and_insert(find_hdr, **kwargs)
            return results
        # Else retrieve it
        else:
            analysis_evs = conn.find_data_reference(
                data_reference_header=dref_hdr)
            if not isinstance(data_names, list):
                data_names = [data_names]
            results = [fsc.retrieve(analysis_evs[ev_idx]['data'][data_name])
                       for data_name in data_names]
            if len(results) == 1:
                results = results[0]
            return results


def run_and_insert(hdr, find_hdr, function_name, run_function):
    # write analysisstore header
    analysis_hdr_uid = conn.insert_analysis_header(
        uid=str(uuid4()),
        time=time.time(),
        provenance={'function_name': function_name})

    # run the analysis function
    try:
        rf = run_function(hdr, find_hdr)

        results, data_keys = run_function(hdr, find_hdr)
    except:
        # Analysis failed!
        conn.insert_analysis_tail(analysis_header=analysis_hdr_uid,
                                  uid=str(uuid4()),
                                  time=time.time(),
                                  exit_status='failure',
                                  error_msg=sys.exc_info()[0])
        # raise a analysis failed error here
        return
    data_hdr = dict(analysis_header=analysis_hdr_uid, data_keys=data_keys,
                    time=time.time(), uid=str(uuid4()))
    data_hdr_uid = conn.insert_data_reference_header(**data_hdr)

    # insert into analysisstore the results
    for res in results:
        data_ref_uid = conn.insert_data_reference(
            data_hdr,
            uid=str(uuid4()),
            time=time.time(),
            data={k: v for k, v in zip(data_names, res)},
            timestamps={}
        )
    # insert analysis tail
    conn.insert_analysis_tail(analysis_header=analysis_hdr_uid,
                              uid=str(uuid4()),
                              time=time.time(), exit_status='success')
    return results


def sample_analysis_function(hdr, **kwargs):
    # extract data from analysisstore or databroker
    data_keys = {'poni': dict(
        source='pyFAI-calib',
        external='FILESTORE:',
        dtype='dict'
    )}
    data_names = 'blob'
    for event in get_events(hdr):
        res = 1 + 1
        # save
        # insert into filestore
        yield res, data_names, data_keys


def calibrate_detector(hdr, **kwargs):
    energy = get_from_analysisstore(hdr,
                                    function_name='calibrate_energy',
                                    run_function=calibrate_energy,
                                    uid_name='energy_calibration_uid',
                                    data_names='energy')
    # find subtracted dark data
    img = get_from_analysisstore(hdr,
                                 function_name='subtract_dark',
                                 run_function=subtract_dark,
                                 uid_name='dark_img_uid',
                                 data_names='img',
                                 fs_override=True)


def subtract_dark(hdr, dark_hdr, img_name='pe1_img', dark_event_idx=-1):
    dark_events = get_events(dark_hdr, fill=True)
    dark_img = islice(dark_events, dark_event_idx)
    events = get_events(hdr, fill=True)
    imgs = [event['data'][img_name] - dark_img for event in events]
    return imgs


def calibrate_energy():
    pass
