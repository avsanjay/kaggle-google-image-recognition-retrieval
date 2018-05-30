from __future__ import print_function
from pyimagesearch.descriptors import DetectAndDescribe
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
import csv

ROOT_DIR = '/mnt/Samsung1'

RETRIVE_DIR = ROOT_DIR + '/output-retrieval/'

RECOGN_DIR = ROOT_DIR + '/output-recognition/'


OUTPUT_FILE = '/mnt/Samsung1/output-keypoint-recognition/train_orb_features.hdf5'

INPUT_FILE = '/mnt/Samsung1/output-keypoint-test/train_orb_features.hdf5'


db_output = h5py.File(OUTPUT_FILE, mode = "r")
db_input = h5py.File(INPUT_FILE, mode = "r")

detector = FeatureDetector_create("ORB")
matcher = cv2.DescriptorMatcher_create("BruteForce")


train_data = pd.read_csv("recognition_train_file_with_location_db_index.csv", index_col=0)
df = train_data.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')

retrieval_data = pd.read_csv("test_file_with_location_db_index.csv", index_col=0)

retrieval_data['classMatch']  = 0

retrieval_data['nearID'] = None
retrieval_data['nearValue'] = None
retrieval_data['farId'] = None
retrieval_data['farValue'] = None

queryimagenumber  = 0

for index, row1 in retrieval_data.iterrows():

    queryimageID = db_input["image_ids"][row1["dbIndex"]]
    (querystart, queryend) = db_input["index"][row1["dbIndex"]]
    queryrows = db_input["features"][querystart:queryend]
    querykps = queryrows[:, :2]
    querydescs = queryrows[:, 2:]

    maxDist = 0
    maxImage = 0
    minDist = 2
    minImage = 0

    print("queryimagenumber is")
    print(queryimagenumber)

    queryimagenumber = queryimagenumber + 1

    #if queryimagenumber >=10000:
     #   break

    targetclassnumber = 0

    for index2, row2 in train_data.iterrows():


        print("targetclass number is ")
        print(targetclassnumber)

        targetclassnumber = targetclassnumber + 1

        if (row2["dbIndex"] >= 1216949):
            continue

        targetimageID = db_output["image_ids"][row2["dbIndex"]]
        (targetstart, targetend) = db_output["index"][row2["dbIndex"]]
        targetrows = db_output["features"][targetstart:targetend]
        targetkps = targetrows[:, :2]
        targetdescs = targetrows[:, 2:]

        v1 = np.unpackbits(np.array(querydescs, dtype = np.uint8))
        v2 = np.unpackbits(np.array(targetdescs, dtype = np.uint8))

        if v1.shape[0]  != v2.shape[0]:
            continue

        #matches = np.sum(np.logical_xor(v1,v2))
        matches = np.count_nonzero(v1 != v2)

        if matches > maxDist:
            maxDist = matches
            maxImage = row2['landmark_id']
        elif matches <= minDist:
            minDist = matches
            minImage = row2['landmark_id']

retrieval_data.to_csv('retrieval_image with class_matches.csv')