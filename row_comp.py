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


detector = FeatureDetector_create("ORB")
matcher = DescriptorMatcher_create("BruteForce")


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

#detector = FeatureDetector_create("ORB")
#matcher = DescriptorMatcher_create("BruteForce")


#train_data = pd.read_csv("train_file_with_dbindex.csv", index_col=0)
#df = train_data.groupby('landmark_id')['id'].nunique().sort_values(ascending = False).reset_index(name = 'count')


#targetimageID = db_output["image_ids"][row2["dbIndex"]]
#(targetstart, targetend) = db_output["index"][row2["dbIndex"]]



queryrows = db_input["features"][:]
querydescs = queryrows[:,2:]

#length2 = querydescs.shape[0]

length_output = db_output["features"].shape[0]

memoryNumber = 200000

for i in range(0, length_output, memoryNumber):

    if i + memoryNumber > length_output:
        targetrows = db_output["features"][i: length_output]
    else:
        targetrows = db_output["features"][ i : i + memoryNumber ]

    targetdescs =  targetrows[:,2:]

    number = i

    #matches = match_descriptors(np.asarray(querydescs, np.float32),
    #                              np.asarray(targetdescs, np.float32), 2)

    rawMatches = matcher.knnMatch(np.asarray(querydescs, np.float32),
                                  np.asarray(targetdescs, np.float32), 2)
    matches = []

    if rawMatches is not None:
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance passes David Lowe's ratio test
            if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        length = len(matches)


    #outfilename1 = 'first' + str(number) + '.npy'

    outfilename1 = 'matches' + str(number) + '.npy'



    #idxfirst = np.flatnonzero(np.in1d(asvoid(targetdescs), asvoid(querydescs)))

    np.save(outfilename1,matches)
    #idxsecond = np.flatnonzero(np.in1d(asvoid(querydescs), asvoid(targetdescs)))

    #outfilename2= 'second' + str(number) + '.npy'

    #np.save(outfilename2, idxsecond)








