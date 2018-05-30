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

OUTPUT_FILE = '/mnt/Samsung2/output-keypoint2-recognition/train_orb_features.hdf5'

INPUT_FILE = '/mnt/Samsung1/output-keypoint-test/retrieval_train_orb_features.hdf5'


db_output = h5py.File(OUTPUT_FILE, mode = "r")
db_input = h5py.File(INPUT_FILE, mode = "r")

#detector = FeatureDetector_create("ORB")
#matcher = DescriptorMatcher_create("BruteForce")


#train_data = pd.read_csv("train_file_with_dbindex.csv", index_col=0)
#df = train_data.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')


#targetimageID = db_output["image_ids"][row2["dbIndex"]]
#(targetstart, targetend) = db_output["index"][row2["dbIndex"]]



queryrows = db_input["features"][:]
querydescs = queryrows[:,2:]

length2 = querydescs.shape[0]



targetrows = db_output["features"][:60000000]
#targetkps = targetrows[:, :2]
targetdescs = targetrows[:, 2:]

length = targetdescs.shape[0]

#firstArray = np.flatnonzero((querydescs == targetdescs).all(1))

i = 0;
result = []

#result = targetdescs.searchsorted(targetdescs)

#for s in querydescs:
#    idx = np.argwhere([np.all((targetdescs-s)==0, axis=1)])[0][1]
#    result.append([i,idx])
#    i = i+1
#    print(i)

#dims = targetdescs.max(0)+1
#out = np.where(np.in1d(np.ravel_multi_index(targetdescs.T,dims),\
                       #np.ravel_multi_index(querydescs.T,dims)))[0]


def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed which treat
    entire rows as one value.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """ Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

idx = np.flatnonzero(np.in1d(asvoid(targetdescs), asvoid(querydescs)))




np.save("first_array.npy", idx)

