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

#OUTPUT_FILE = OUTPUT_DIR + 'train_orb_features.hdf5'

OUTPUT_FILE = '/mnt/Samsung2/output-keypoint2/train_orb_features.hdf5'


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



train_data = pd.read_csv("../train_file_with_location_full.csv", index_col=0)

#df = train_data.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')

#train_data['dbIndex'] = train_data.index

train_data['dbIndex'] = 0

#db to dataframe index



for i in range ( db["image_ids"].shape[0]):
    #print(i)
    imageId = db["image_ids"][i]
    #print(imageId)
    #index6 = train_data.index[train_data.id == imageId]
    #print(index6)
    #print(train_data.id[train_data.id == imageId].index.tolist())
    index7 = train_data.id[train_data.id == imageId].index.tolist()
    #print("index7")
    print(index7)
    #print(index7[0])
    train_data.at[index7[0],'dbIndex'] = i
    #index21 = train_data.index[train_data['id'] == imageId].tolist()
    #train_data.loc[index21[0]]["dbIndex"] = i

train_data.to_csv('train_file_with_dbindex.csv')