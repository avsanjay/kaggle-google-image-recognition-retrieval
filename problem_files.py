# USAGE
# python index_features.py --features-db output/train_orb_features.hdf5

# import the necessary packages
from __future__ import print_function
from imutils.feature import FeatureDetector_create, DescriptorExtractor_create
from imutils import paths
import argparse
import imutils
import cv2
import pandas as pd
import numpy as np
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

#df = pd.read_csv('/home/sanjay-titan/PycharmProjects/recognition/train_file_with_location.csv')

#full_ids = df['filelocation']
#ids = df['id']

# initialize the keypoint detector, local invariant descriptor, and the descriptor
# pipeline
#detector = FeatureDetector_create("ORB")
#descriptor = DescriptorExtractor_create("ORB")
#dad = DetectAndDescribe(detector, descriptor)

# initialize the feature indexer
#fi = FeatureIndexer(args["features_db"], estNumImages=args["approx_images"],#
#	maxBufferSize=args["max_buffer_size"], verbose=True)



imagePath = "/mnt/Samsung1/output-recognition/008a9d55dbc26097.jpg"
image = cv2.imread(imagePath)

cv2.imshow('image',image)
#cv2.waitkey(0)

image = imutils.resize(image, width=320)
array_alpha = np.array([1.25])
array_beta = np.array([-100.0])

cv2.add(image,array_beta, image)
cv2.multiply(image, array_alpha, image)

maxIntensity = 255.0
phi = 1
theta = 1


newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
newImage0 = np.array(newImage0)

cv2.imshow('newimage0', newImage0)
#cv2.waitkey(0)

newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
newImage1 = np.array(newImage1)

cv2.imshow('newImage1', newImage1)

	# describe the image
	#(kps, descs) = dad.describe(image)

	# if either the keypoints or descriptors are None, then ignore the image
	#if kps is None or descs is None:
	#	continue


    # scikit



descriptor_extractor = ORB(n_keypoints=500)
descriptor_extractor.detect_and_extract(newImage1)
kps = descriptor_extractor.keypoints
descs = descriptor_extractor.descriptors
