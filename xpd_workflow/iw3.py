from databroker import db, get_events, get_table
import numpy as np
import scipy.stats as sts
from skbeam.diffraction import twotheta_to_q, bin_edges_to_centers
from skbeam.core.utils import radius_to_twotheta
from skbeam.core.mask import margin_mask, ring_blur_mask
from itertools import islice
from diffpy.pdfgetx import PDFGetter
import pyFAI
from skbeam.core.utils import q_to_twotheta
import subprocess as sbp
from filestore import commands as fsc
from tifffile import imsave
from analysisstore.client.commands import AnalysisClient
import time
from uuid import uuid4

conf = dict(host='xf28id-ca1.cs.nsls2.local', port=7767)
conn = AnalysisClient(conf)
# workflow for x-ray scattering
default_pdf_params = {'qmin': 0.0, 'qmax': 25., 'qmaxinst': 30, 'rpoly': .9}


# 0. do calibration
def run_calibration(event, detector, w_or_e, w_or_e_val):
    function_kwargs = locals()
    del function_kwargs['event']

    img = subs_dark(event, dark_hdr_idx, img_key)
    # Put image on disk as a tiff somewhere
    imsave('{}_{}_dk_subs.tif'.format(event['descriptor']['run_start']['uid']),
           event['seq_num'])
    # change dirs to analysis dir
    sbp.run(['pyFAI-calib', '-D {} -{} {} -c {} {}'.format(
        detector, w_or_e, w_or_e_val, calibration_file, image_file
    )])
    # add files to filestore
    # add entry to analysisstore
    md = dict(run_header_uid=event['descriptor']['run_start_header']['uid'],
              seq_num=event['seq_num]'],
              event_uid=event['uid'],
              function='run_calibration',
              function_kwargs=function_kwargs)
    a_hdr_uid = conn.insert_analysis_header(time=time.time(), uid=str(uuid4()),
                                            provenance=prov_kwargs,
                                            **md)
    data_keys = {'poni': dict(source='pyFAI-calib', external='FILESTORE:',
                              dtype='dict')}

    data_hdr = dict(analysis_header=a_hdr_uid, data_keys=data_keys,
                    time=time.time(), uid=str(uuid4()))
    data_hdr_uid = conn.insert_data_reference_header(**data_hdr)
    data_ref_uid = conn.insert_data_reference(data_hdr, uid=str(uuid4()),
                                              time=time.time(),
                                              data={'poni': ''},
                                              timestamps={})
    conn.insert_analysis_tail(analysis_header=a_hdr_uid, uid=str(uuid4()),
                              time=time.time(), exit_status='success')
    geo = fsc.retrieve()
    return geo


# 1. associate with a calibration
def get_calibration(event, cal_hdr_idx=-1, cal_file=None):
    """
    Get calibration file out of analysisstore
    Parameters
    ----------
    event: mds.event
        The event to be processed
    cal_hdr_idx: int, optional
        The integer which specifies which calibration to use if multiple are
        specified for an event. Defaults to the latest calibration
    cal_file: str, optional
        An override for analysisstore, the path to a .poni file

    Returns
    -------
    geo: pyFAI.geometry
        The geometry in for the calibration

    """
    if cal_file is None:
        return pyFAI.load(cal_file)
    cal_uid = event['descriptor']['run_start']['calibration_uid']
    cal_hdr = db(calibration_uid=cal_uid, is_calibration=True)[cal_hdr_idx]
    # Get geo from analysisstore associated with cal_hdr
    a_hdrs = conn.find_analysis_header(run_header_uid=cal_hdr)
    if not a_hdrs:
        run_calibration(event, *args)
    else:
        dref_hdr = conn.find_data_reference_header(analysis_header=a_hdrs[0])
        # check if calibration exit_status good
        calib_evs = conn.find_data_reference(data_reference_header=dref_hdr)
        # Filestore magic on events goes here
    geo = 1
    return geo


# 2a. subtact dark
def subs_dark(event, dark_hdr_idx=-1, img_key='pe1_image'):
    """
    Subtract a dark from the foreground data
    Parameters
    ----------
    event; mds.event
        The event to be processed
    dark_hdr_idx: int, optional
        The integer which specifies which dark to use if multiple are
        specified for an event. Defaults to the latest dark
    img_key: str, optional
        The key for the image in the event['data'] dictionary, defaults to
        `pe1_image`

    Returns
    -------
    dark_img: array
        The dark image associated with the event

    """
    dark_uid = event['descriptor']['run_start']['dark_uid']
    dark_hdr = db(dark_uid=dark_uid, role='dark')[dark_hdr_idx]
    dark_events = get_events(dark_hdr, fill=True)
    # TODO: allow for indexing of the data so we can get back different darks
    dark_img = next(dark_events)['data'][img_key]
    return event['data'][img_key] - dark_img


# 2b. - 5. process to I(Q)

def single_mask_integration(img, geo,
                            alpha=2.5,
                            pol=.95,
                            lower_thresh=0.0,
                            upper_thresh=None,
                            margin=30.,
                            tmsk=None,
                            statistics='median',
                            ):
    """
    Performs polarization correction, masking, and integration for an image
    Parameters
    ----------
    img: 2darray
        The image
    geo: pyFAI.geometry
        The calibrtion geometry for the detector
    alpha: float or tuple of float or array, optional
        The allowed standard deviation multiplier, defaults to 2.5
        `alpha * std - mean < x < alpha * std + mean` will remain
    pol: float, optional
        The polarization correction, defaults to .95, if None no correctin will
        be applied
    lower_thresh: float, optional
        Threshold for a lower threshold mask, defaults to 0.0, if None no lower
        threshold is applied
    upper_thresh: float, optional
        Threshold for an upper threshold mask, defaults to None, if None no
         higher threshold is applied
    margin: int, optional
        The margin size for a margin mask, defaults to 30 pixels
    tmsk: bool array. optional
        The starting mask, defaults to None, if None, a clean mask is used as
        the starting mask
    statistics: str or function or list of str and/or functions
        Statistical measure applied to the rings

    Returns
    -------
    q: array
        The Q bins
    iq: array or list of arrays
        The resulting integrated statistics, if multiple statistics are
        specified, multiple arrays are returned in a list
    tmsk: bool array
        The mask used in the integration

    """
    # Correct for polarization
    if pol:
        img /= geo.polarization(img.shape, pol)

    r = geo.rArray(img.shape)
    pixel_size = [getattr(geo, a) for a in ['pixel1', 'pixel2']]
    rres = np.hypot(*pixel_size)
    rbins = np.arange(np.min(r) - rres / 2., np.max(r) + rres / 2., rres)

    # Pre masking data

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

    fmsk_img = img[tmsk]
    fmsk_r = r[tmsk]

    # Post masking data
    qbins = twotheta_to_q(radius_to_twotheta(.23, rbins), .143)
    bs_kwargs = {'bins': rbins}
    bs_args = (fmsk_r, fmsk_img)

    integrated_statistics = []
    if not isinstance(statistics, 'list'):
        statistics = [statistics]
    for stat in statistics:
        integrated_stat = sts.binned_statistic(*bs_args, statistic=stat,
                                               **bs_kwargs)[0]
        integrated_statistics.append(integrated_stat)

    if len(integrated_statistics) == 1:
        integrated_statistics = integrated_statistics[0]

    return bin_edges_to_centers(qbins), integrated_statistics, tmsk


def process_to_iq(event, img_key='pe1_image',
                  archived_analysis=False, **kwargs):
    """
    Process an event to I(Q)

    Parameters
    ----------
    event: mds.event
        The event to be processed
    img_key: str, optional
        The key for the image in the event['data'] dictionary, defaults to
        `pe1_image`
    kwargs:
        key word arguments to `single_mask_integration`

    Returns
    -------

    """
    # Pull data from analysisstoreis
    if archived_analysis and event in []:
        pass
    else:
        geo = get_calibration(event=event)
        q, iq, msk = single_mask_integration(event['data'][img_key], geo,
                                             **kwargs)
    tth = q_to_twotheta(q, geo.wavelength)
    # save everything to filestore
    return q, iq, tth, msk


def get_background(event, match_keys=None, bg_hdr_idx=-1):
    """

    Parameters
    ----------
    event: mds.event
        The event to be processed
    match_keys: list of str, optional
        The keys to match between the foreground and background defaults to
        None. If None, the latest background image is used
    bg_hdr_idx: int
        The background header to be used, if multiple are associated with the
        data

    Returns
    -------
    event:
        The background event most suited to the foreground event
    """
    # Or pull this out of amostra
    bg_uid = event['descriptor']['run_start']['background_uid']
    bg_hdr = db(background_uid=bg_uid, is_background=True)[bg_hdr_idx]
    # Match up the foreground event with the background
    if match_keys is not None:
        table = get_table(bg_hdr, fields=match_keys)
        # This logic could get more complex if multiple fields are to be matched
        bg_event_idx = np.argmin(np.abs(event[match_keys] - table[match_keys]))
    else:
        bg_event_idx = 0
        pass
    bg_event = next(islice(get_events(bg_hdr), bg_event_idx))
    return bg_event


def background_subtract(event, bg_scale=1, match_keys=None, bg_hdr_idx=-1,
                        img_key='pe1_image', **kwargs):
    """

    Parameters
    ----------
    event: mds.event
        The event to be processed
    bg_scale: float, optional
        The scale factor for the background, defaults to 1
    match_keys: list of str, optional
        The keys to match between the foreground and background defaults to
        None. If None, the latest background image is used
    bg_hdr_idx: int
        The background header to be used, if multiple are associated with the
        data
    img_key: str, optional
        The key for the image in the event['data'] dictionary, defaults to
        `pe1_image`
    kwargs:
        key word arguments to `single_mask_integration`

    Returns
    -------
    q: 1darray
        The Q values
    corr_iq: 1darray
        The background subtracted I(Q)
    """
    q, iq, msk = process_to_iq(event=event, img_key=img_key, **kwargs)
    bg_event = get_background(event, match_keys, bg_hdr_idx)
    bg_q, bg_iq, bg_msk = get_background(bg_event, img_key=img_key, **kwargs)
    # TODO: optimize the background subtraction
    corr_iq = iq - bg_iq * bg_scale
    return q, corr_iq


def ripple_func(gr, window=3):
    w = gr - np.convolve(gr, np.ones(window) / window, 'same')
    return np.sum(np.abs(w))


def opt_qmax(q, iq, params, opt_func, opt_func_kwargs={},
             min_qmax=25, resolution=.1,
             pdf_getter=PDFGetter()):
    """
    Find the optimal qmax and qmaxinst for FFT

    Parameters
    ----------
    q: np.array
        The Q array
    iq: np.array
        The I(Q) arrau
    params: dict
        The base PDF parameters
    opt_func: func
        The function to optimize, lower values result in better PDFs
    opt_func_kwargs: dict
        kwargs to the optimzation function
    min_qmax: float
        Minimum bound for Qmax
    resolution: float
        Stepsize for Qmax and Qmaxinst
    pdf_getter: PDFGetter
        The PDF generating function

    Returns
    -------
    r: array
        Inter-atomic distance array
    gr: np.array
        The PDF
    params: dict
        The parameters used to produce the PDF
    """
    a = np.arange(min_qmax, q[-1], resolution)
    ripple_array = np.zeros((len(a), len(a))) * np.nan
    # TODO: may want to parallelize this to make it faster
    for i in range(len(a)):
        params['qmax'] = a[i]
        params['rstep'] = np.pi / params['qmax']
        for j in range(i, len(a)):
            params['qmaxinst'] = a[j]
            r, gr = pdf_getter(q, iq, **params)
            ripple_array[i, j] = opt_func(gr, **opt_func_kwargs)
    b1, b2 = np.unravel_index(np.nanargmin(ripple_array), ripple_array.shape)
    params['qmax'] = b1
    params['qmaxinst'] = b2
    params['rstep'] = np.pi / params['qmax']
    r, gr = pdf_getter(q, iq, **params)
    return r, gr, params


def process_to_pdf(event, match_keys=None, bg_hdr_idx=-1, img_key='pe1_image',
                   pdf_params=default_pdf_params, opt_func=ripple_func,
                   opt_func_kwargs={'window': 3}, min_qmax=25, resolution=.1,
                   pdf_getter=PDFGetter(),
                   **kwargs):
    q, corr_iq = background_subtract(event, match_keys, bg_hdr_idx, img_key,
                                     **kwargs)
    r, gr, params = opt_qmax(q, corr_iq, pdf_params, opt_func, opt_func_kwargs,
                             min_qmax, resolution, pdf_getter)
    return r, gr, params
