from __future__ import division, print_function

import os

import matplotlib.pyplot as plt
import scipy.stats as sts
from diffpy.pdfgetx import PDFGetter

from sidewinder_spec.utils.handlers import *
from skbeam.diffraction import bin_edges_to_centers
from skbeam.io.save_powder_output import save_output
from skbeam.core.utils import twotheta_to_q


def mask_edge(img_shape, edge_size):
    """
    Mask the edge of an image

    Parameters
    -----------
    img_shape: tuple
        The shape of the image
    edge_size: int
        Number of pixels to mask from the edge
    Returns
    --------
    2darray:
        The mask array, bad pixels are 0
    """
    mask = np.zeros(img_shape, dtype=bool)
    mask[edge_size:-edge_size, edge_size:-edge_size] = True
    return mask


def generate_q_bins(rmax, pixel_size, distance, wavelength, rmin=0):
    """
    Generate the Q bins at the resolution of the detector
    Parameters
    -----------
    rmax: float
        The maximum radial distance on the detector in distance units.
        Note that this should go to the bottom edge of the pixel.
    pixel_size: float
        The size of the pixels, in the same units as rmax
    distance: float
        The sample to detector distance, in the same units as rmax
    wavelength: float
        The wavelength of the x-rays
    rmin: float, optional
        The minimum radial distance on the detector in distance units. Defaults
        to zero. Note that this should be the bottom of the pixel

    Returns
    -------
    ndarray:
        The bin edges, suitable for np.histogram or
        scipy.stats.binned_statistic
    """
    pixel_bottom = np.arange(rmin, rmax, pixel_size)
    pixel_top = pixel_bottom + pixel_size

    bottom_tth = np.arctan(pixel_bottom[0] / distance)
    top_tth = np.arctan(pixel_top / distance)

    top_q = twotheta_to_q(top_tth, wavelength)

    bins = np.zeros(len(top_q) + 1)

    bins[0] = twotheta_to_q(bottom_tth, wavelength)
    bins[1:] = top_q
    return bins


def ring_blur_mask(img, q, alpha, bins, mask=None):
    """
    Perform a annular mask, which checks the ring statistics and masks any
    pixels which have a value greater or less than alpha * std away from the
    mean
    Parameters
    ----------
    img: 2darray
        The  image
    q: 2darray
        The  array which maps pixels to Q space
    alpha: float or tuple or, 1darray
        Then number of acceptable standard deviations, if tuple then we use
        a linear distribution of alphas from alpha[0] to alpha[1], if array
        then we just use that as the distribution of alphas
    rmax: float
        The maximum radial distance on the detector
    pixel_size: float
        The size of the pixels, in the same units as rmax
    distance: float
        The sample to detector distance, in the same units as rmax
    wavelength: float
        The wavelength of the x-rays
    mask: 1darray
        A starting flattened mask
    Returns
    --------
    2darray:
        The mask
    """

    if mask is None:
        mask = np.ones(img.shape).astype(bool)
    if mask.shape != img.shape:
        mask = mask.reshape(img.shape)
    msk_img = img[mask]
    msk_q = q[mask]

    int_q = np.zeros(q.shape, dtype=np.int)
    for i in range(len(bins) - 1):
        t_array = (bins[i] <= q) & (q < bins[i + 1])
        int_q[t_array] = i - 1
    # integration
    mean = sts.binned_statistic(msk_q, msk_img, bins=bins[1:],
                                statistic='mean')[0]
    std = sts.binned_statistic(msk_q, msk_img, bins=bins[1:],
                               statistic=np.std)[0]
    if type(alpha) is tuple:
        alpha = np.linspace(alpha[0], alpha[1], len(std))
    threshold = alpha * std
    lower = mean - threshold
    upper = mean + threshold

    # single out the too low and too high pixels
    too_low = img < lower[int_q]
    too_hi = img > upper[int_q]

    mask = mask * ~too_low * ~too_hi
    return mask.astype(bool)


def single_mask_integration(img, geo, alpha=(3., 3.), tmsk=None,
                            statistics='median'):
    # Correct for polarization
    img /= geo.polarization(img.shape, .95)

    r = geo.rArray(img.shape)
    q = geo.qArray(img.shape) / 10  # pyFAI works in nm**-1, we want A**-1
    bins = generate_q_bins(np.max(r) - .5 * geo.pixel1,
                           geo.pixel1, geo.dist, geo.wavelength*10**10)
    # Pre masking data
    bs_kwargs = {'bins': bins,
                 # 'range': [0, fq.max()]
                 }

    if tmsk is None:
        tmsk = np.ones(img.shape, dtype=int).astype(bool)
    tmsk *= mask_edge(img.shape, 30)
    tmsk *= ring_blur_mask(img, q, alpha, bins, mask=tmsk)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True)
    img2 = img.copy()
    img2[~tmsk] = 0.0

    ax1.imshow(img)
    ax2.imshow(img2)
    ax3.imshow(~tmsk)

    fmsk_img = img[tmsk]
    fmsk_q = q[tmsk]

    # Post masking data
    bs_args = (fmsk_q, fmsk_img)

    integrated_statistics = []
    if hasattr(statistics, '__iter__'):
        for stat in statistics:
            integrated_stat = sts.binned_statistic(*bs_args, statistic=stat,
                                                   **bs_kwargs)[0]
            integrated_statistics.append(integrated_stat)
    else:
        integrated_stat = sts.binned_statistic(*bs_args, statistic=statistics,
                                               **bs_kwargs)[0]
        integrated_statistics.append(integrated_stat)

    return bin_edges_to_centers(bins), integrated_statistics, (tmsk)


def single_event_workflow(foreground_args,
                          background_args,
                          temp=True,
                          post_processing=('IQ', 'PDF'),
                          dir_path=None,
                          fn_stem=None,
                          pdf_dict=None,
                          plot=False,
                          save=False,
                          ):
    # Unpack the dataframes
    (fg_img, fg_i0, fg_md, geo) = foreground_args
    (bg_img, bg_i0, bg_md, geo) = background_args
    # Mask/Integrate the foreground
    fg_x, fg_y, _ = single_mask_integration(fg_img, geo)
    fg_y /= fg_i0 / fg_md['summedexposures']
    # Mask/Integrate the background
    bg_x, bg_y, _ = single_mask_integration(bg_img, geo)
    bg_y /= bg_i0 / bg_md['summedexposures']
    # Subtract the background
    bg_subed_y = fg_y - bg_y

    # PDFgetx3
    if 'PDF' in post_processing and np.max(fg_x) >= 25.:
        z = PDFGetter()
        if pdf_dict is None:
            pdf_dict = [{'qmin': 1.5,
                         'qmax': 25., 'qmaxinst': 25.,
                         'rpoly': .9,
                         'rmax': 40.,
                         'composition': 'Ni', 'dataformat': 'Qnm',
                         }]
        elif not isinstance(pdf_dict, (tuple, list)):
            pdf_dict = [pdf_dict]
        for pd in pdf_dict:
            r, gr = z(fg_x, bg_subed_y, **pd)
            rgr = np.vstack((r, gr)).T
            qfq = np.vstack(z.fq).T
            if plot:
                f, (ax1, ax2, ax3) = plt.subplots(3, 1)
                ax1.plot(fg_x, fg_y, 'b-')
                ax1.plot(fg_x, bg_y, 'g-')
                ax1.plot(fg_x, bg_subed_y, 'r--')
                ax2.plot(z.fq[0], z.fq[1])
                ax3.plot(r, gr)
            if fn_stem is not None and dir_path is not None and save:
                for data, end in zip([rgr, qfq], ['gr', 'fq']):
                    np.savetxt(os.path.join(dir_path,
                                            '{}.{}'.format(fn_stem, end)), rgr)
            save_output(fg_x, bg_subed_y, fn_stem, q_or_2theta='Q',
                        dir_path=dir_path)

    if 'IQ' in post_processing and fn_stem is not None and dir_path is not None and save:
        save_output(fg_x, bg_subed_y, fn_stem, q_or_2theta='Q',
                    dir_path=dir_path)
    plt.show()
    return

if __name__ == '__main__':
    plt.style.use('/mnt/bulk-data/Masters_Thesis/config/thesis.mplstyle')

    from pyFAI import load
    from pims import TiffStack
    base = '/mnt/bulk-data/Dropbox/BNL_Project/misc/CGO_summed/CGO_summed'
    # geo = Geometry()
    geo = load(os.path.join(base, 'Ni_stnd', 'Ni_STD_60s-00004.poni'))
    # Geometry.load(os.path.join(base, 'ni_std', 'Ni_STD_60s-00004.poni'))
    img = TiffStack(os.path.join(base, 'Sample1_350um', 'Sample_1_CGO_4_0_19_summed.tif'))[0]

    x, int_sts,(mask) = single_mask_integration(img, geo, statistics=('mean', 'median', np.std), alpha=(3, 3))
    fig, ax = plt.subplots()
    ax.plot(x, int_sts[0])
    ax.plot(x, int_sts[1])
    fig, ax = plt.subplots()
    ax.plot(x, int_sts[2]/int_sts[1])
    plt.show()
