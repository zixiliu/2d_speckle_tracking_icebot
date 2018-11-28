import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb

def find_blob(filename, plotit = False): 

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


	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

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
	im_with_keypoints = cv2.drawKeypoints(im_ori, keypoints, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	processed = cv2.drawKeypoints(ultra8, keypoints, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	plt.figure(figsize=(20,8))
	plt.subplot(121)
	plt.imshow(im_with_keypoints)
	plt.subplot(122)
	plt.imshow(processed)
	plt.suptitle(filename)
	
	if plotit:
		plt.show()
	else: 
		# pdb.set_trace()

		folder_name = 'img/'

		if filename[-7] == '_':
			save_file_name = folder_name+'0'+filename[-6:-3]+'png'
		elif filename[-6] == '_':
			save_file_name = folder_name+'00'+filename[-5:-3]+'png'
		else:
			save_file_name = folder_name+filename[-7:-3]+'png'

		plt.savefig(save_file_name)