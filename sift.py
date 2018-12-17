import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import pdb
import glob


## Load Image
mypath = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/"
filename = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/16062017_125453973_4050.jpg"
file1 = filename
filename = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/16062017_125454009_4051.jpg"
file2 = filename

def helper_strip_img_text(filename):

	a = cv.imread(filename)

	imgray = a.copy()
	imgray = cv.cvtColor(imgray,cv.COLOR_BGR2GRAY)

	######################################
	### Mask Out Info by Contour
	######################################
	imgray_cp = imgray.copy()
	ret,thresh = cv.threshold(imgray_cp,5,255,0)
	thresh = cv.erode(thresh, None, iterations=2)
	thresh = cv.dilate(thresh, None, iterations=2)

	image, contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	c = max(contours, key=cv.contourArea)


	other_info = imgray.copy()
	other_info = cv.fillPoly(other_info, pts =[c], color=(255,255,255))

	ultra_ori = imgray - other_info
	return ultra_ori

def print_matching_keypoints(file1, file2, i):

	print(file1)

	img1 = helper_strip_img_text(file2) # queryImage
	img2 = helper_strip_img_text(file1) # trainImage


	# Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	# BFMatcher with default params
	bf = cv.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	# Apply ratio test
	good = []
	idx_train = []
	idx_query = []

	for m,n in matches:
	    if m.distance < 0.75*n.distance:
	        good.append(m)
	        idx_train.append(m.trainIdx)
	        idx_query.append(m.queryIdx)
	# cv.drawMatchesKnn expects list of lists as matches.

	# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good, img2, flags=2)

	green = (0, 255, 0)

	img=cv.drawKeypoints(img1,np.array(kp1)[idx_query],img1, color=green,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv.imwrite('sift/sift_keypoints_file'+str(i)+'.jpg',img)


	# img=cv.drawKeypoints(img2,np.array(kp2)[idx_train],img2, color =(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# cv.imwrite('sift_keypoints_file2.jpg',img)

def print_keypoints(file, i):


	print(file)

	img = cv.imread(file)
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	sift = cv.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None)
	img=cv.drawKeypoints(gray,kp,img)
	cv.imwrite('sift_kp/'+str(i)+'.jpg',img)




def process_frames(file_path):

	files = glob.glob(file_path+"*.jpg")
	files = sorted(files)
	start_idx = 4050
	file_of_interest = files[start_idx:start_idx+30]

	for i in range(len(file_of_interest)-1):
		this_file = file_of_interest[i]
		next_file = file_of_interest[i+1]

		# print_matching_keypoints(this_file, next_file, i)
		print_keypoints(this_file,i+start_idx)
