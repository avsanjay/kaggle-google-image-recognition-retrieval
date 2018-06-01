from __future__ import print_function
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

ROOT_DIR = '/mnt/Samsung1'

RETRIVE_DIR = ROOT_DIR + '/output-retrieval'

RECOGN_DIR = ROOT_DIR + '/output-recognition/'

OUTPUT_DIR = ROOT_DIR + '/output-keypoint/'

#OUTPUT_FILE = OUTPUT_DIR + 'train_orb_features.hdf5'

OUTPUT_FILE = '/mnt/Samsung2/output-keypoint2-recognition/train_orb_features.hdf5'


db = h5py.File(OUTPUT_FILE, mode = "r")

list(db.keys())

print(db["image_ids"].shape)
print(db["image_ids"])
print(db["image_ids"].value)


imageid_db_lis = db["image_ids"]
#print(db["image_ids"][8])
print(db["index"].shape)

print(db["features"].shape)

detector = FeatureDetector_create("ORB")
matcher = DescriptorMatcher_create("BruteForce")



train_data = pd.read_csv("train_file_with_dbindex.csv", index_col=0)

df = train_data.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')



train_data['nearID'] = None
train_data['nearValue'] = None
train_data['farId'] = None
train_data['farValue'] = None

df2 = pd.DataFrame(np.array(
    df),index = df.index)

df2.columns = ['landmark_id', 'count']

j = 0

trainClassDict = {}





for index, row1 in df2.iterrows():
    if row1['count'] >= 100 and row1['count'] <= 200:

        print("in the first for loop")
        print(" landmark id is ")
        print(row1['landmark_id'])
        print("count is")
        print(row1['count'])
        j = j +1
        print(" i have been in the first loop")
        print(j)

        classCount =row1['count']


        df_landmarkId =  train_data.loc[train_data['landmark_id'] == row1['landmark_id'] ]
        df_landmarkCopy = df_landmarkId.copy()
        #imageIDList = db["image_ids"][:]
        #df_duplandmarKId = df_landmarkId

        maxDist = 0
        maxImage = 0
        minDist = 2
        minImage = 0


        targetlist = np.zeros([1, 32])


        for index2, row2 in df_landmarkId.iterrows():

            print("in the second for loop")

            queryimageID = db["image_ids"][row2["dbIndex"]]
            (querystart, queryend) = db["index"][row2["dbIndex"]]
            queryrows = db["features"][querystart:queryend]
            querykps = queryrows[:, :2]
            querydescs = queryrows[:, 2:]

            print(" landmark id is ")
            print(row1['landmark_id'])
            print("Total number of images in the class is")
            print(row1['count'])
            print("the current image withinthe class which is being considered is")
            classCount = classCount - 1
            print(classCount)
            print(" out of")
            print(row1['count'])
            imageCount = row1['count']

            targetimageID = db["image_ids"][row2["dbIndex"]]
            (targetstart, targetend) = db["index"][row2["dbIndex"]]
            targetrows = db["features"][targetstart:targetend]
            targetkps = targetrows[:, :2]
            targetdescs = targetrows[:, 2:]



            targetlist = np.vstack((targetlist, targetdescs))

            print("i am end of third loop")

        trainClassDict[row1['landmark_id']] = targetlist

        rawMatches = matcher.knnMatch(np.asarray(querydescs, np.float32), np.asarray(trainClassDict[row1['landmark_id']], np.float32), 2)
        print("i finished doing the raw matches")
        matches = []

        if rawMatches is not None:
            # loop over the raw matches
            for m in rawMatches:
                # ensure the distance passes David Lowe's ratio test
                if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                    matches.append((m[0].trainIdx, m[0].queryIdx))

            length = len(matches)


df_final = pd.DataFrame.from_dict(trainClassDict, orient="index")

df_final.to_csv("dict_data.csv")
