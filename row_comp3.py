from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import cv2
import pandas as pd
import numpy as np
import h5py
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create
import skimage
from skimage import data
from skimage.feature import ( match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import scipy.sparse

ROOT_DIR = '/mnt/Samsung1'

ROOT_DIR2 = '/mnt/Samsung2'

RETRIVE_DIR = ROOT_DIR + '/output-retrieval'

RECOGN_DIR = ROOT_DIR + '/output-recognition/'

OUTPUT_DIR = ROOT_DIR + '/output-keypoint/'

OUTPUT_DIR2 = ROOT_DIR2 + '/output-numpy/'

#OUTPUT_FILE = OUTPUT_DIR + 'train_orb_features.hdf5'

OUTPUT_FILE = '/mnt/Samsung2/output-keypoint2-recognition/train_orb_features.hdf5'

INPUT_FILE = '/mnt/Samsung1/output-keypoint-test/retrieval_train_orb_features.hdf5'


db_output = h5py.File(OUTPUT_FILE, mode = "r")
db_input = h5py.File(INPUT_FILE, mode = "r")

memoryNumber_query = 10000
memoryNumber_target = 10000


targetlist = np.zeros([1, 32])
querylist = np.zeros([1,32])


targetBreakNumber = 0
queryBreakNumber = 0

length_output = db_output["features"].shape[0]
length_input = db_input["features"].shape[0]


#for  j in range(0, length_input, memoryNumber_target):

    #targetBreakNumber =  targetBreakNumber + 1

    #if targetBreakNumber > 10:
    #    break

#    if j + memoryNumber_target > length_input:
#        queryrows = db_input["features"][j:length_input]
#    else:
#        queryrows = db_input["features"][j: j+10]

#    querydescs = queryrows[:, 2:]
#    querylist = np.vstack((querylist, querydescs))


#np.save("querylist.npy",querylist)

#for i in range(0, length_output, memoryNumber_query):

    #targetBreakNumber = targetBreakNumber + 1

    #if targetBreakNumber > 10:
    #   break

#    if i + memoryNumber_query > length_output:
#        targetrows = db_output["features"][i: length_output]
#    else:
#         targetrows = db_output["features"][ i : i + 10 ]

#    targetdescs =  targetrows[:,2:]
#    targetlist = np.vstack((targetlist, targetdescs))

#np.save("targetlist.npy",targetlist)

targetlist = np.load("targetlist.npy")
querylist = np.load("querylist.npy")

length_output1 = targetlist.shape[0]
length_input1 = querylist.shape[0]




for  j in range(0, length_input1, memoryNumber_query):

    if j + memoryNumber_query > length_input1:
        queryrows = querylist[j:length_input1]
    else:
        queryrows = querylist[j: memoryNumber_query]

    querydescs = queryrows[:, 2:]

    for i in range(0, length_output1, memoryNumber_target):

        if i + memoryNumber_query > length_output1:
            targetrows = targetlist[i: length_output1]
        else:
            targetrows = targetlist[ i : memoryNumber_target]

        targetdescs =  targetrows[:,2:]

        number1 = j
        number2 = i

        n_x, n_d  = querydescs.shape
        n_y, n_d = targetdescs.shape

        values, ix = np.unique(np.vstack((querydescs,targetdescs)),return_inverse=True)
        n_unique = len(values)

        ix_hat = ix.reshape(-1, n_d)
        ix_x_hat = ix_hat[:n_x]
        ix_y_hat = ix_hat[n_x:]

        x_hat = scipy.sparse.lil_matrix((n_x, n_unique), dtype=int)
        x_hat[np.arange(n_x)[:,None], ix_x_hat]= 1

        y_hat = scipy.sparse.lil_matrix((len(targetdescs), len(values)), dtype=int)
        y_hat[np.arange(n_y)[:,None], ix_y_hat]= 1

        matches = x_hat.dot(y_hat.T)


        k = 6

        outfilename1 = OUTPUT_DIR2 + 'matches' + str(number1) + str(number2) + '.npy'

        np.save(outfilename1,matches)

numberofRows = number1
numberofColumns = number2
combinedArray = np.zeros(shape=(numberofRows,numberofColumns))


for  j in range(0, length_input1, memoryNumber_query):



    for i in range(0, length_output1, memoryNumber_target):


        number1 = j
        number2 = i

        outfilename2 = OUTPUT_DIR2 + 'matches' + str(number1) + str(number2) + '.npy'


        combinedArray[number1][number2] = np.load(outfilename2)




