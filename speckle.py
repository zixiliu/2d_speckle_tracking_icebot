import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb
import glob

### Global variables
dis_thresh = 490



def helper_save_my_image(filename, ultra_w_circle, processed_w_circle, plotit, ext=None):
	# if processed_w_circle == None:
	# 	plt.figure(figsize=(20,8))
	# 	plt.imshow(ultra_w_circle)
	# 	plt.title(filename)
	# else:
	plt.figure(figsize=(20,8))
	plt.subplot(121)
	plt.imshow(ultra_w_circle)
	plt.subplot(122)
	plt.imshow(processed_w_circle)
	plt.suptitle(filename)

	if plotit:
		plt.show()
	else:
		# pdb.set_trace()

		folder_name = 'img5/'

		if ext == None:
			if filename[-7] == '_':
				save_file_name = folder_name+'0'+filename[-6:-4]+'.png'
			elif filename[-6] == '_':
				save_file_name = folder_name+'00'+filename[-5:-4]+'.png'
			else:
				save_file_name = folder_name+filename[-7:-4]+'.png'
		else:
			if filename[-7] == '_':
				save_file_name = folder_name+'0'+filename[-6:-4]+ext+'.png'
			elif filename[-6] == '_':
				save_file_name = folder_name+'00'+filename[-5:-4]+ext+'.png'
			else:
				save_file_name = folder_name+filename[-7:-4]+ext+'.png'
		plt.savefig(save_file_name)

def helper_get_distance_sq(kp1, kp2):
	'''Get distance between two keypoints'''
	(x1, y1) = kp1.pt
	(x2, y2) = kp2.pt
	return ((x1-x2)**2 + (y1-y2)**2)

def helper_find_next_speckle(keypoints_frame1, keypoints_frame2):
	global dis_thresh

	start_pts = []
	end_pts = []

	for kp1 in keypoints_frame1:
		# find points in frame 2 that are close to this point in frame 1
		close_pts = []
		for kp2 in keypoints_frame2:
			if helper_get_distance_sq(kp1, kp2) <= dis_thresh:
				close_pts.append(kp2)
		# find the point withint the closest points of the most similar area
		area_diff = np.Inf
		if len(close_pts) > 0:
			the_pt = close_pts[0]
			the_dis = np.abs(the_pt.size - kp1.size)
			for pts in close_pts[1::]:
				if np.abs(pts.size - kp1.size) < the_dis:
					the_pt = pts

			start_pts.append(kp1)
			end_pts.append(the_pt)

	return (start_pts, end_pts)


def find_blob(filename, saveit = False, plotit = False):

	a = cv2.imread(filename)
	# pdb.set_trace()

	imgray = a.copy()
	imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)

	######################################
	### Mask Out Info by Contour
	######################################
	imgray_cp = imgray.copy()
	ret,thresh = cv2.threshold(imgray_cp,5,255,0)
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# if plotit:
	# 	plt.figure(figsize=(15,15))
	# 	plt.imshow(thresh,  cmap='gist_gray')
	# 	plt.title('Thresh after threshold imgray')
	# 	plt.show()

	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = max(contours, key=cv2.contourArea)

	# img_cp = a.copy()
	# if plotit:
	# 	plt.figure(figsize=(15, 6))
	# 	plt.subplot(121)
	# 	cv2.drawContours(img_cp, c, -1, (0,255,0), 5)
	# 	plt.imshow(img_cp)
	# 	plt.subplot(122)
	# 	cv2.drawContours(img_cp, contours, -1, (0,255,0), 5)
	# 	plt.imshow(img_cp)
	# 	plt.show()

	other_info = imgray.copy()
	other_info = cv2.fillPoly(other_info, pts =[c], color=(255,255,255))

	ultra_ori = imgray - other_info
	ultra = ultra_ori.copy()

	# if plotit:
	# 	plt.figure(figsize=(15,15))
	# 	plt.imshow(ultra,  cmap='gist_gray')
	# 	plt.title('Gray Scale Image')
	# 	plt.show()

	######################################
	### Pre-processing
	######################################

	## Normalize ultra
	ultra = ultra / (ultra.max() - ultra.min())*255
	#print(ultra.shape, ultra.max(), ultra.min())

	# if plotit:
	# 	plt.figure(figsize=(15, 15))
	# 	plt.imshow(ultra_ori, cmap='gist_gray')
	# 	plt.title('Normalized Image')
	# 	plt.show()

	######################################
	### Find Speckle
	######################################
	ret,thresh = cv2.threshold(ultra,140,255,0)


	thresh = cv2.erode(thresh, None, iterations=1)
	thresh = cv2.dilate(thresh, None, iterations=1)

	ultra8 = (255 - thresh).astype('uint8')

	# plt.figure()
	# plt.imshow(ultra8)
	# plt.show()

	# Set up the detector with default parameters.
	detector = cv2.SimpleBlobDetector_create()

	# Detect blobs.
	keypoints = detector.detect(ultra8)

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_ori = ultra_ori.copy()
	ultra_w_circle = cv2.drawKeypoints(im_ori, keypoints, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	processed_w_circle = cv2.drawKeypoints(ultra8, keypoints, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	if saveit:
		helper_save_my_image(filename, ultra_w_circle, processed_w_circle, plotit)

	# pdb.set_trace()

	return keypoints, ultra_ori, ultra8


def two_frames(file1, file2):
	keypoints_frame1, ultra1 = find_blob(file1)
	keypoints_frame2, ultra2 = find_blob(file2)

	(start_pts, end_pts) = helper_find_next_speckle(keypoints_frame1, keypoints_frame2)


	im_ori1 = ultra1.copy()
	ultra_w_circle1 = cv2.drawKeypoints(im_ori1, start_pts, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	helper_save_my_image(file1, ultra_w_circle1, None, False,'_s')


	im_ori2 = ultra2.copy()
	ultra_w_circle2 = cv2.drawKeypoints(im_ori2, end_pts, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	helper_save_my_image(file2, ultra_w_circle2, None, False,'_e')


def three_frames(file1, file2, file3):
	keypoints_frame1, ultra1, _ = find_blob(file1)
	keypoints_frame2, ultra2, ultra8 = find_blob(file2)
	keypoints_frame3, ultra2, _ = find_blob(file3)

	(start_pts1, end_pts1) = helper_find_next_speckle(keypoints_frame1, keypoints_frame2)
	(start_pts2, end_pts2) = helper_find_next_speckle(keypoints_frame2, keypoints_frame3)
	# keypoints = end_pts1 + start_pts2

	im_ori2 = ultra2.copy()
	ultra_w_circle2 = cv2.drawKeypoints(im_ori2, end_pts1, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	ultra_w_circle2 = cv2.drawKeypoints(ultra_w_circle2, start_pts2, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	ultra8 = cv2.drawKeypoints(ultra8, end_pts1+start_pts2, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	for i in range(len(start_pts1)):
		a = (int(start_pts1[i].pt[0]), int(start_pts1[i].pt[1]))
		b = (int(end_pts1[i].pt[0]), int(end_pts1[i].pt[1]))
		cv2.line(ultra_w_circle2,a, b,(255,255,0),2)
		cv2.line(ultra8,a, b,(255,0,0),2)
	# for i in range(len(start_pts2)):
	# 	a = (int(start_pts2[i].pt[0]), int(start_pts2[i].pt[1]))
	# 	b = (int(end_pts2[i].pt[0]), int(end_pts2[i].pt[1]))
	# 	cv2.line(ultra_w_circle2,a, b,(255,0,0),1)

	helper_save_my_image(file2, ultra_w_circle2, ultra8, False)

def process_frames(file_path):

	files = glob.glob(file_path+"*.jpg")
	files = sorted(files)
	file_of_interest = files[0:100]

	for i in range(1, len(file_of_interest)-1):
		prev_file = file_of_interest[i-1]
		this_file = file_of_interest[i]
		next_file = file_of_interest[i+1]

		# two_frames(this_file, next_file)
		three_frames(prev_file, this_file, next_file)


