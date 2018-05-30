from __future__ import print_function

import argparse
import cv2
import pandas as pd
import numpy as np
import h5py
import skimage
from skimage import data
from skimage.feature import ( match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import defaultdict

#length2 = querydescs.shape[0]

length_output = 60000000

landMarkDict = defaultdict(list)

count = 0

for i in range(0, length_output, 10000000):

    if count > 40:
       break


    number = i

    outfilename1 = 'first' + str(number) + '.npy'

    idxfirst = np.load(outfilename1)

    outfilename2= 'second' + str(number) + '.npy'

    idxsecond = np.load( outfilename2)

    indArryLen = len(idxfirst)

    secArryLen = len(idxsecond)

    if indArryLen >= secArryLen:
        length = secArryLen
    else:
        length = indArryLen

    for j in range(length):

            key = idxsecond[j]//500
            if i is not 0:
               value = (idxfirst[j]//500)  +( i // 500)
            else:
               value = (idxfirst[j]//500)

            landMarkDict[key].append(value)


    count = count + 1

print("i am done with the loops")