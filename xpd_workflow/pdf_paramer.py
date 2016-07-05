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

fg_x, bg_subed_y = loadData('/media/christopher/5TB_Backup/deepthought/bulk-data/research_data/USC_beamtime/APS_March_2016/S1/temp_exp/00001.chi', unpack=True)
pd = {'qmin': 1.5,
             'qmax': 25., 'qmaxinst': 25.,
             'rpoly': .9,
             'rmax': 40.,
             'composition': 'Pr2NiO4', 'dataformat': 'QA',
             }
a = np.arange(25, 35,  .1)
ripple_aray = np.zeros((len(a), len(a)))
for ei, i in enumerate(a):
    for ej, j in enumerate(a):
        if i <= j:
            pd['qmaxinst'] = j
            pd['qmax'] = i
            z = PDFGetter()
            r, gr = z(fg_x, bg_subed_y, **pd)
            w = gr - np.convolve(gr, np.ones(3)/3, 'same')
            ripple_sum = np.sum(abs(w))
            ripple_aray[ei, ej] = ripple_sum
plt.imshow(ripple_aray, cmap='viridis')
plt.colorbar()
plt.show()