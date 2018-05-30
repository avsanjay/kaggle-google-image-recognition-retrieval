# USAGE
# python index_features.py --features-db output/train_orb_features.hdf5

# import the necessary packages
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
import tqdm
import skimage
from skimage import data
from skimage.feature import ( match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


ROOT_DIR = '/mnt/Samsung1'

RETRIVE_DIR = ROOT_DIR + '/output-retrieval'

RECOGN_DIR = ROOT_DIR + '/output-recognition/'

OUTPUT_DIR = ROOT_DIR + '/output-keypoint/'

OUTPUT_FILE = '/mnt/Samsung2/output-keypoint2/train_orb_features.hdf5'

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset",
#	help="Path to the directory that contains the images to be indexed")
#ap.add_argument("-f", "--features-db", required=True,
#	help="Path to where the features database will be stored")
#ap.add_argument("-a", "--approx-images", type=int, default=500,
#	help="Approximate # of images in the dataset")
#ap.add_argument("-b", "--max-buffer-size", type=int, default=117000,
#	help="Maximum buffer size for # of features to be stored in memory")
#args = vars(ap.parse_args())

df = pd.read_csv('/home/sanjay-titan/PycharmProjects/recognition/train_file_with_location.csv')

full_ids = df['filelocation']
ids = df['id']

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
detector = FeatureDetector_create("ORB")
descriptor = DescriptorExtractor_create("ORB")
dad = DetectAndDescribe(detector, descriptor)

# initialize the feature indexer
#fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],#
#	maxBufferSize=args["max_buffer_size"], verbose=True)

fi = FeatureIndexer(OUTPUT_FILE, estNumImages=500,
	           maxBufferSize=117000, verbose=True)

# loop over the images in the dataset
for (i, imagePath) in enumerate(full_ids):
	# check to see if progress should be displayed
	if i > 0 and i % 10 == 0:
		fi._debug("processed {} images".format(i), msgType="[PROGRESS]")

	# extract the image filename (i.e. the unique image ID) from the image
	# path, then load the image itself
	filename = ids[i]
	print("filename")
	print(filename)

	# cv2

	if imagePath == "008a9d55dbc26097":
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=320)
		array_alpha = np.array([1.25])
		array_beta = np.array([-100.0])

		cv2.add(image,array_beta, image)
		cv2.multiply(image, array_alpha, image)
	else:

	    image = cv2.imread(imagePath)
	    #image = imutils.resize(image, width=320)
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# describe the image
	#(kps, descs) = dad.describe(image)

	# if either the keypoints or descriptors are None, then ignore the image
	#if kps is None or descs is None:
	#	continue


    # scikit

	descriptor_extractor = ORB(n_keypoints=500)
	k = descriptor_extractor.detect_and_extract(image)
	if k == -1:
		continue
	kps = descriptor_extractor.keypoints
	descs = descriptor_extractor.descriptors

	if kps is None or descs is None:
		continue



	# index the features
	fi.add(filename, kps, descs)

# finish the indexing process
fi.finish()
