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

ROOT_DIR = '/mnt/Samsung1'

RETRIVE_DIR = ROOT_DIR + '/output-retrieval'

RECOGN_DIR = ROOT_DIR + '/output-recognition/'

OUTPUT_DIR = ROOT_DIR + '/output-keypoint/'

OUTPUT_FILE = OUTPUT_DIR + 'train_orb_features.hdf5'


db = h5py.File(OUTPUT_FILE, mode = "r")

list(db.keys())

print(db["image_ids"].shape)
print(db["image_ids"])
#print(db["image_ids"].value)
print(db["image_ids"][8])
print(db["index"].shape)

print(db["features"].shape)

detector = FeatureDetector_create("ORB")
matcher = DescriptorMatcher_create("BruteForce")



train_data = pd.read_csv("../train_file_with_location.csv", index_col=0)

df = train_data.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')

train_data['dbIndex'] = train_data.index

#db to dataframe index



#for i in range ( db["image_ids"].shape[0]):
#    #print(i)
#    imageId = db["image_ids"][i]
#    print(imageId)
    #index6 = train_data.index[train_data.id == imageId]
    #print(index6)
    #print(train_data.id[train_data.id == imageId].index.tolist())
#    index7 = train_data.id[train_data.id == imageId].index.tolist()
#    print("index7")
#    print(index7)
#    print(index7[0])
#    train_data.at[index7[0],'dbIndex'] = i
    #index21 = train_data.index[train_data['id'] == imageId].tolist()
    #train_data.loc[index21[0]]["dbIndex"] = i


#train_data.reindex(columns = [*train_data.columns.tolist(), 'nearID', 'nearValue', 'farID', 'farValue'], fill_value = 0)

train_data['nearID'] = None
train_data['nearValue'] = None
train_data['farId'] = None
train_data['farValue'] = None

df2 = pd.DataFrame(np.array(
    df),index = df.index)

df2.columns = ['landmark_id', 'count']

for index, row1 in df2.iterrows():
    if row1['count'] >= 100:

        print("in the first for loop")
        df_landmarkId =  train_data.loc[train_data['landmark_id'] == row1['landmark_id'] ]
        df_landmarkCopy = df_landmarkId.copy()
        #imageIDList = db["image_ids"][:]
        #df_duplandmarKId = df_landmarkId

        maxDist = 0
        maxImage = 0
        minDist = 2
        minImage = 0


        for index2, row2 in df_landmarkId.iterrows():

            print("in the second for loop")

            queryimageID = db["image_ids"][row2["dbIndex"]]
            (querystart, queryend) = db["index"][row2["dbIndex"]]
            queryrows = db["features"][querystart:queryend]
            querykps = queryrows[:, :2]
            querydescs = queryrows[:, 2:]

            for index3, row3 in df_landmarkId.iterrows():

                print("in the third for loop")

                if row3["dbIndex"] == row2["dbIndex"]:
                    continue

                targetimageID = db["image_ids"][row3["dbIndex"]]
                (targetstart, targetend) = db["index"][row3["dbIndex"]]
                targetrows = db["features"][targetstart:targetend]
                targetkps = targetrows[:, :2]
                targetdescs = targetrows[:, 2:]


                #rawMatches = matcher.knnMatch(querydescs, targetdescs, 2)

                #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                # create BFMatcher object
                #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                #querydescsexp = querydescs[0]
                #targetdescsexp = targetdescs[0]

                # Match descriptors.
                #matches = bf.match(querydescsexp, targetdescsexp)

                matches = match_descriptors(querydescs,targetdescs, cross_check=True)

                length = matches.shape[0]

                if length > maxDist:
                    maxDist = length
                    maxImage = row3["id"]
                elif length <= minDist:
                    minDist = length
                    minImage = row3["id"]

            # the 3rd for loop has ended

            train_data.at[index2, 'nearID'] = maxImage
            train_data.at[index2, 'nearValue'] = maxDist
            train_data.at[index2, 'farID'] = minImage
            train_data.at[index2, 'farValue'] = minDist


train_data.to_csv('train_file_with_class_wise_near_distant_matches.csv')





                # Sort them in the order of their distance.
                #matches = sorted(matches, key=lambda x: x.distance)




                #matches = []

                #if rawMatches is not None:
                #    # loop over the raw matches
                #    for m in rawMatches:
                #        # ensure the distance passes David Lowe's ratio test
                #        if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                #            matches.append((m[0].trainIdx, m[0].queryIdx))










