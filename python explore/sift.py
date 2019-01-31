import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pdb
import glob



## Load Image
mypath = "images/exp5_images/"
file1 = mypath+"16062017_125258857_600.jpg"
file2 = mypath+'16062017_125258889_601.jpg'

def helper_strip_img_text(filename):

	a = cv.imread(filename)

	imgray = a.copy()
	imgray = cv.cvtColor(imgray,cv.COLOR_BGR2GRAY)

	### Mask Out Info by Contour
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
	# return imgray

def helper_getLength(kp1, kp2): 
	x1, y1 = kp1.pt[0], kp1.pt[1]
	x2, y2 = kp2.pt[0], kp2.pt[1]
	return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def is_the_same_kp(kp1, kp2): 
	if (kp1.pt[0] == kp2.pt[0]) and (kp1.pt[1] == kp2.pt[1]) and (kp1.size == kp2.size):
		return True
	else:
		return False

def helper_find_matches(query_img, train_img, dis_thresh): 

	# Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1s, des1 = sift.detectAndCompute(query_img,None)
	kp2s, des2 = sift.detectAndCompute(train_img,None)
	# pdb.set_trace()
	# BFMatcher with default params
	bf = cv.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Apply ratio test
	good = []
	idx_train = []
	idx_query = []
	for ms in matches:
		if ms[0].distance > 0.75* ms[1].distance:
			distances = []
			for m in ms: 
				kp1 = kp1s[m.queryIdx]
				kp2 = kp2s[m.trainIdx]
				dis = helper_getLength(kp1, kp2)
				distances.append(dis)
			idx = np.argmin(np.array(distances))
			m = ms[idx]
		else:
			m = ms[0]
		kp1 = kp1s[m.queryIdx]
		kp2 = kp2s[m.trainIdx]
		dis = helper_getLength(kp1, kp2)
		if dis < dis_thresh:	 # 30 for consecutive
			good.append([m])
			idx_train.append(m.trainIdx)
			idx_query.append(m.queryIdx)

	## No Constraint:  
	# for ms in matches:
	# 	m = ms[0]
	# 	good.append([m])
	# 	idx_train.append(m.trainIdx)
	# 	idx_query.append(m.queryIdx)
	
	# cv.drawMatchesKnn expects list of lists as matches.
	# img3 = cv.drawMatchesKnn(img1,kp1s,img2,kp2s,good, img2, flags=2)
	# print("Number of matches: ",len(matches),", Number of features: ",  len(good))

	return kp1s, kp2s, idx_train, idx_query

	

def print_matching_keypoints(this_file, prev_file, next_file, file_num, foldername, dis_thresh=30):

	print(this_file)
	prev_img = helper_strip_img_text(prev_file) 
	this_img = helper_strip_img_text(this_file) 
	next_img = helper_strip_img_text(next_file) 
	
	this_kpts_wrt_prev, prev_kps, idx_train_prev, idx_query_prev = helper_find_matches(this_img, prev_img, dis_thresh=dis_thresh)

	this_kpts_wrt_next, next_kps, idx_train, idx_query_next = helper_find_matches(this_img, next_img, dis_thresh=dis_thresh)

	
	red = (0,0,255)
	blue = (255,0,0)
	green = (0,255,0)
	yellow = (255, 255, 0)
	light_blue = (0,255,255)

	r, c = this_img.shape

	# superimpose = cv.imread(this_file)
	# superimpose[:,:,0] = this_img
	# superimpose[:,:,1] = np.zeros([r,c])
	# superimpose[:,:,2] = prev_img


	### Plot keypoints in this frame that are matched with the previous frame
	#super impose:
	# img=cv.drawKeypoints(superimpose,np.array(this_kpts_wrt_prev)[idx_query_prev],superimpose, color=light_blue,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	img=cv.drawKeypoints(this_img,np.array(this_kpts_wrt_prev)[idx_query_prev],this_img, color=light_blue,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# ### Plot keypoints and their matched keypoints in the next frame
	# # img=cv.drawKeypoints(img,np.array(next_kps)[idx_train],this_img, color=(0,255,0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	## Draw a line between this keypoint to it in the next frame
	for i in range(len(idx_train)):
		i1 = idx_query_next[i]
		i2 = idx_train[i]		
		a = (int(this_kpts_wrt_next[i1].pt[0]), int(this_kpts_wrt_next[i1].pt[1]))
		b = (int(next_kps[i2].pt[0]), int(next_kps[i2].pt[1]))
		cv.line(img,a, b,green,1)

	## Draw a line between this keypoint to it in the previous frame
	for i in range(len(idx_train_prev)):
		i1 = idx_query_prev[i]
		i2 = idx_train_prev[i]		
		a = (int(this_kpts_wrt_prev[i1].pt[0]), int(this_kpts_wrt_prev[i1].pt[1]))
		b = (int(prev_kps[i2].pt[0]), int(prev_kps[i2].pt[1]))
		cv.line(img,a, b,light_blue, 1)
	
	img=cv.drawKeypoints(img,np.array(this_kpts_wrt_next)[idx_query_next],img, color=green,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# ## Overlapping keypoints
	# overlapping_kps = []
	# # pdb.set_trace()
	# for kp1 in np.array(this_kpts_wrt_next)[idx_query_next]:
	# 	for kp2 in np.array(this_kpts_wrt_prev)[idx_query_prev]:
	# 		if is_the_same_kp(kp1, kp2):
	# 			overlapping_kps.append(kp1)
	# 			# print("overlap!")

	# img=cv.drawKeypoints(img,np.array(overlapping_kps),img, color=green,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv.imwrite(foldername+str(file_num)+'.jpg',img)


def print_keypoints(file, i):
	print(file)
	img = cv.imread(file)
	gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	sift = cv.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None)
	img=cv.drawKeypoints(gray,kp,img)
	cv.imwrite('images/sift_kp/'+str(i)+'.jpg',img)


def process_consecutive_frames(file_path):

	files = glob.glob(file_path+"*.jpg")
	files = sorted(files)
	start_idx = 0
	file_of_interest = files[start_idx:start_idx+30]
	# pdb.set_trace()

	foldername = 'images/sift/'

	for i in range(1, len(file_of_interest)-1):
		prev_file = file_of_interest[i-1]
		this_file = file_of_interest[i]
		next_file = file_of_interest[i+1]

		print_matching_keypoints(this_file, prev_file, next_file,  i+4050,foldername, dis_thresh=30)
		# print_keypoints(this_file,i+start_idx)

def process_same_frames(file_path):

	files = glob.glob(file_path+"*.jpg")
	files = sorted(files)
	#start_idx = 0
	file_of_interest = [int(round(x)) for x in np.arange(0, len(files), 18.46)]

	foldername = 'images/sift_same/'

	for i in range(1, len(file_of_interest)-1):
		prev_file = files[file_of_interest[i-1]]
		this_file = files[file_of_interest[i]]
		next_file = files[file_of_interest[i+1]]
		print_matching_keypoints(this_file, prev_file, next_file,  this_file[-8:-3], foldername, dis_thresh=10)
		# print_keypoints(this_file,i+start_idx)
