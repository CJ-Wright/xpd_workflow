'''
Copyright (c) 2014 Brookhaven National Laboratory All rights reserved.
Use is subject to license terms and conditions.
@author: Christopher J. Wright'''
__author__ = 'Christopher J. Wright'

import matplotlib.pyplot as plt
import os
import scipy.signal as signal
from diffpy.pdfgetx import PDFGetter, loadData
import numpy as np

cvk = 3
# for cvk in range(1, 6):
qmaxes = []
qmaxints = []
ks = np.arange(0, 165, 10)
rgrs = []
for k in ks:
    fg_x, bg_subed_y = loadData(
        '/media/christopher/5TB_Backup/deepthought/bulk-data/research_data/USC_beamtime/APS_March_2016/S1/temp_exp/{}.chi'.format(str(k).zfill(5)),
        unpack=True)
    # fg_x, bg_subed_y = loadData('/media/christopher/5TB_Backup/deepthought/bulk-data/research_data/USC_beamtime/APS_March_2016/S1/temp_exp/d25_S6_VT-00000.chi', unpack=True)
    # fg_x /= 10
    pd = {'qmin': 1.5,
          'qmax': 26., 'qmaxinst': 35.,
          'rpoly': .9,
          'rmax': 40.,
          'composition': 'Pr2NiO4', 'dataformat': 'QA',
          }
    z = PDFGetter()
    a = np.arange(25, fg_x[-1], .1)
    ripple_array = np.zeros((len(a), len(a)))
    ripple_list = []
    print(ripple_array.size)
    '''
    for i in a:
        pd['qmax'] = i
        print(i)
        pd['rstep'] = np.pi/pd['qmax']
        r, gr = z(fg_x, bg_subed_y, **pd)
        w = gr - np.convolve(gr, np.ones(3) / 3, 'same')
        ripple_sum = np.sum(abs(w))
        ripple_list.append(ripple_sum)
    plt.plot(a, ripple_list)
    plt.show()
    AAA
    # '''
    for ei, i in enumerate(a):
        for ej, j in enumerate(a):
            if i <= j:
                pd['qmaxinst'] = j
                pd['qmax'] = i
                pd['rstep'] = np.pi / pd['qmax']
                r, gr = z(fg_x, bg_subed_y, **pd)
                w = gr - np.convolve(gr, np.ones(cvk) / cvk, 'same')
                ripple_sum = np.sum(abs(w))
                ripple_array[ei, ej] = ripple_sum
            else:
                ripple_array[ei, ej] = np.nan
    # plt.imshow(ripple_array, cmap='viridis', interpolation='none', extent=[a[0], a[-1], a[0], a[-1]], origin='lower left',
    #            vmax=1.1 * np.nanmin(ripple_array), aspect='auto')
    # plt.colorbar()
    # plt.xlabel('Qmaxinst')
    # plt.ylabel('Qmax')
    b1, b2 = np.unravel_index(np.nanargmin(ripple_array), ripple_array.shape)
    print(a[b1], a[b2], np.nanmin(ripple_array))
    plt.show()
    pd['qmax'] = a[b1]
    pd['qmaxinst'] = a[b2]
    qmaxes.append(a[b1])
    qmaxints.append(a[b2])
    pd['rstep'] = .01
    r, gr = z(fg_x, bg_subed_y, **pd)
    rgrs.append((r, gr))
    # plt.plot(r, gr)
    # plt.show()
for r, gr in rgrs:
    plt.plot(r, gr)
plt.show()
