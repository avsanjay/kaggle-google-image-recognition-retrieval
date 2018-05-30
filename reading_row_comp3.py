from pyimagesearch.indexer import FeatureIndexer
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

memoryNumber_query =  10000
memoryNumber_target = 10000


targetlist = np.zeros([1, 32])
querylist = np.zeros([1,32])


targetBreakNumber = 0
queryBreakNumber = 0

length_output = db_output["features"].shape[0]
length_input = db_input["features"].shape[0]


length_output1 = (length_output//50)//10000
length_input1 = (length_input//50)//10000




for  j in range(length_input1):


    for i in range(length_output1):


        number1 = j
        number2 = i * 10000

        outfilename1 = OUTPUT_DIR2 + 'matches' + str(number1) + str(number2) + '.npy'

        readArray =  np.load(outfilename1)

        k = 6