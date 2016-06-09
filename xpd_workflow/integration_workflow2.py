from __future__ import division, print_function

from databroker import db, get_events
from datamuxer import DataMuxer
from metadatastore.api import db_connect as mds_db_connect

from filestore.api import db_connect as fs_db_connect
from filestore.api import retrieve
from xpd_workflow.mask_tools import *
from xpd_workflow.single_event_integration import single_event_workflow

fs_db_connect(
    **{'database': 'data-processing-dev', 'host': 'localhost', 'port': 27017})
mds_db_connect(
    **{'database': 'data-processing-dev', 'host': 'localhost', 'port': 27017})

pdf_dict_list = [
    # {'qmin': 1.5,
    #  'qmax': 25., 'qmaxinst': 25.,
    #  'rpoly': .9,
    #  'rmax': 40.,
    #  'composition': 'Pr2NiO4', 'dataformat': 'QA',
    #  },
    {'qmin': 1.5,
     'qmax': 28., 'qmaxinst': 30,
     'rpoly': .9,
     'rmax': 40.,
     'composition': 'Pr2NiO4', 'dataformat': 'QA',
     },
    # {'qmin': 1.5,
    #  'qmax': 35., 'qmaxinst': 35.,
    #  'rpoly': .9,
    #  'rmax': 40.,
    #  'composition': 'Pr2NiO4', 'dataformat': 'QA',
    #  }
]


def main(plot=True, super_plot=False):
    # Get headers of interest
    hdrs = db(
        run_folder='/mnt/bulk-data/research_data/USC_beamtime/APS_March_2016/S1/temp_exp'
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

        bg_idx = None
        for i, a in b.iterrows():
            print('start event {}'.format(i))
            if 'T' in df.keys():
                (i0, T, detz, img, md) = a
            else:
                (i0, detz, img, md) = a
            cal_idx = np.argmin((detz - cal_dists) ** 2)
            geo = geos[cal_idx]
            fg_args = (retrieve(img), i0, md, geo)

            if 'T' in df.keys():
                # Get background signal at correct temperature (or closest to)
                temp_metric = np.abs(T - bg_binned['T'].values)

                # Throw out the ones not at the PDF distance
                temp_metric[bg_binned['detz'].values != detz] = 1e6
                bg_idx2 = np.argmin(temp_metric)
                if bg_idx is None or bg_idx != bg_idx2:
                    bg_idx = bg_idx2
                    bg_args = (retrieve(bg_binned.img.values[bg_idx][0]),
                               bg_binned['I0'].values[bg_idx],
                               bg_binned.metadata.values[bg_idx][0],
                               geo)

            single_event_workflow(fg_args, bg_args,
                                  # plot=True,
                                  pdf_dict=pdf_dict_list,
                                  dir_path=hdr['start']['run_folder'],
                                  fn_stem=str(i).zfill(5),
                                  save=True
                                  )
            print('end event {}'.format(i))

if __name__ == '__main__':
    main()
    # main_queue()
    exit()
