from __future__ import division, print_function

import matplotlib.pyplot as plt
from databroker import db, get_events
from datamuxer import DataMuxer
from diffpy.pdfgetx import PDFGetter
from metadatastore.api import db_connect as mds_db_connect

from filestore.api import db_connect as fs_db_connect
from filestore.api import retrieve
from sidewinder_spec.utils.handlers import *
from skbeam.core.mask import *
from xpd_workflow.mask_tools import *
from skbeam.io.save_powder_output import save_output
import os

fs_db_connect(
    **{'database': 'data-processing-dev', 'host': 'localhost', 'port': 27017})
mds_db_connect(
    **{'database': 'data-processing-dev', 'host': 'localhost', 'port': 27017})


def main(plot=True, super_plot=False):
    # Get headers of interest
    hdrs = db(
        # run_folder='/mnt/bulk-data/research_data/USC_beamtime/APS_March_2016/S1/temp_exp'
        # is_calibration=False
        # is_calibration=True
    )
    # Get the background header and mux it's events
    bg_hdr = db(
        run_folder='/mnt/bulk-data/research_data/USC_beamtime/APS_March_2016/'
                   'Quartz_Background/temp_exp')

    bg_dm = DataMuxer()
    bg_dm.append_events(get_events(bg_hdr))
    bg_binned = bg_dm.bin_on('img', interpolation={'T': 'linear'})

    for hdr in hdrs:
        print(hdr['start']['run_folder'], hdr['start']['uid'])

        # Get calibrations
        if not hdr['start']['is_calibration']:
            cals = [db[u]['start']['poni'][0] for u in
                    hdr['start']['calibration']]
        else:
            cals = [p for p in hdr['start']['poni']]

        geos = [retrieve(p) for p in cals]
        cal_dists = np.asarray(
            [g.dist for g in geos]) * 100  # convert to meters

        # Get starting masks
        # start_masks = [retrieve(p) for p in hdr['start']['mask']]

        # Give the datamuxer our data

        dm = DataMuxer()
        dm.append_events(get_events(hdr))
        df = dm.to_sparse_dataframe()
        binned = dm.bin_on('img', interpolation={'T': 'linear'})

        if 'T' in df.keys():
            b = binned[['I0', 'T', 'detz', 'img', 'metadata']]
        else:
            b = binned[['I0', 'detz', 'img', 'metadata']]

        total_mask_dict = {}
        bg_idx = None
        for i, a in b.iterrows():
            print('start event {}'.format(i))
            if 'T' in df.keys():
                (i0, T, detz, img, md) = a
            else:
                (i0, detz, img, md) = a

            # Find the correct calibration file
            cal_idx = np.argmin((detz - cal_dists) ** 2)
            geo = geos[cal_idx]
            img /= geo.polarization(img.shape, .95)

            r = geo.rArray(img.shape)
            q = geo.qArray(img.shape)

            bins = 2000

            # Pre masking data
            bs_kwargs = {'statistic': (np.median,),
                         'bins': bins,
                         'range': [0, q.max()]}

            # If we don't have a mask for that distance make one
            if detz not in total_mask_dict.keys():
                if detz < 30.:
                    alpha = (4.5, 2.)
                else:
                    alpha = (4.5, 2.)
                # total_mask = start_masks[cal_idx]
                total_mask_dict[detz] = np.ones(img.shape, dtype=bool)
                total_mask_dict[detz] *= mask_edge(img.shape, 30)
                total_mask_dict[detz] *= ring_blur_mask(img, r, geo.pixel1,
                                                        alpha,
                                                        mask=total_mask_dict[
                                                            detz])
                total_mask = total_mask_dict[detz]
                if super_plot:
                    plt.imshow(total_mask, interpolation='None')
                    plt.show()
            else:
                total_mask = total_mask_dict[detz]

            # Post masking data
            median, x = binstats(q[total_mask], img[total_mask],
                                 **bs_kwargs)

            # Normalize by number of subframes and I0
            # print('normalize by', i0 * md['summedexposures'])
            y = median[0] / i0 / md['summedexposures']
            # y = median[0]

            if 'T' in df.keys():
                # Get background signal at correct temperature (or closest to)
                temp_metric = np.abs(T - bg_binned['T'].values)

                # Throw out the ones not at the PDF distance
                temp_metric[bg_binned['detz'].values != detz] = 1e6
                bg_idx2 = np.argmin(temp_metric)

                # Only reload a file if needed
                if bg_idx is None or bg_idx != bg_idx2 or bg_chi is None:
                    bg_idx = bg_idx2
                    bg_img = bg_binned.img.values[bg_idx][0]

                    bg_fmsk_q = q[total_mask]
                    bg_fmsk_img = bg_img[total_mask]
                    bs_args = (bg_fmsk_q, bg_fmsk_img)
                    bg_chi, x = binstats(*bs_args, **bs_kwargs)
                    # Normalize by number of subframes and I0
                    # print('normalize bg by', bg_binned['I0'].values[bg_idx] * \
                    #          bg_binned.metadata.values[bg_idx][0]['summedexposures'])
                    bg_chi = bg_chi[0] / bg_binned['I0'].values[bg_idx] / \
                             bg_binned.metadata.values[bg_idx][0][
                                 'summedexposures']
                    # bg_chi = bg_chi[0]

            else:
                # Get the background image
                # bg_chi, x = binstats(*bs_args, **bs_kwargs)
                pass

            max_bg_idx = np.argmax(bg_chi)
            # bg_scale = y[max_bg_idx]/bg_chi[max_bg_idx]
            bg_scale = 1.
            bg_y = bg_chi * bg_scale
            # bg_y = i0 * bg_chi / bg_binned.I0.values[bg_idx] * 1.5
            bg_sub_y = y - bg_y
            if detz > 30.:
                print('max intensity', np.max(bg_sub_y))
                print('I0', i0)
                print('summed images', md['summedexposures'])
                print('bg i0', bg_binned['I0'].values[bg_idx])
                print('bg summed images',
                      bg_binned.metadata.values[bg_idx][0]['summedexposures'])

            if detz < 30.:
                if not plot:
                    save_output(x, bg_sub_y, str(i).zfill(5), q_or_2theta='Q',
                                dir_path=hdr['start']['run_folder'])
                z = PDFGetter()
                d1 = {'qmin': 1.5,
                      # 'qmax': 29.5, 'qmaxinst': 29.5,
                      'qmax': 34., 'qmaxinst': 34.,
                      'rpoly': .9,
                      'rmax': 40.,
                      'composition': 'Pr2NiO4', 'dataformat': 'QA',
                      }

                r, gr = z(x, bg_sub_y, **d1)
                rgr = np.vstack((r, gr)).T
                qfq = np.vstack(z.fq).T
                if plot:
                    f, (ax1, ax2, ax3) = plt.subplots(3, 1)
                    ax1.plot(x, y, 'b-')
                    ax1.plot(x, bg_y, 'g-')
                    ax1.plot(x, y - bg_y, 'r--')
                    ax2.plot(z.fq[0], z.fq[1])
                    ax3.plot(r, gr)
                    plt.show()
                if not plot:
                    np.savetxt(os.path.join(hdr['start']['run_folder'],
                                            str(i).zfill(5) + '.gr'), rgr)
                    np.savetxt(os.path.join(hdr['start']['run_folder'],
                                            str(i).zfill(5) + '.fq'), qfq)
            else:
                if plot:
                    plt.plot(x, bg_sub_y)
                    plt.show()
                else:
                    save_output(x, bg_sub_y, str(i).zfill(5), q_or_2theta='Q',
                                dir_path=hdr['start']['run_folder'])
            print('end event {}'.format(i))


if __name__ == '__main__':
    main()
    exit()
