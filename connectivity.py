import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb
import glob

import colorsys
N = 100
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
print(RGB_tuples)


def draw_contour_by_dilation(dilation_filename, ori_filename): 
    # Read the image you want connected components of
    src = cv.imread(dilation_filename)
    ori = cv.imread(ori_filename)
 
    imgray = src.copy()
    # Threshold it so it becomes binary
    imgray = cv.cvtColor(imgray,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    img, contours, hierarchy = cv.findContours(thresh,cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse = True)
    # print(len(contours))
    pdb.set_trace()
    plt.figure()
    for i, c in enumerate(contours):        
        color = (255- 20*i, 40*i % 255, 10*i)
        cv.drawContours(ori, contours, i, color, 2)
    plt.imshow(ori)
    # plt.show()
    # plt.imsave('temp/'+dilation_filename[-8:-1], ori)
    plt.savefig("contour/"+dilation_filename[-8::])



dilation_path = 'dilation/'
ori_path = 'exp5_images/'

dilation_files = glob.glob(dilation_path+"*.jpg")
dilation_files = sorted(dilation_files)

ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

for i in range(30):
    draw_contour_by_dilation(dilation_files[i], ori_files[i])


# sift = cv.xfeatures2d.SIFT_create()
# kp1s, des1 = sift.detectAndCompute(imgray, None)
# img=cv.drawKeypoints(imgray,kp1s,imgray, color=(0,255,0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# pdb.set_trace()
# plt.figure()
# plt.imshow(src)
# plt.show()

# # import the necessary packages
# from imutils import contours
# from skimage import measure
# import numpy as np
# import argparse
# import imutils


# # perform a connected component analysis on the thresholded
# # image, then initialize a mask to store only the "large"
# # components
# labels = measure.label(thresh, neighbors=8, background=0)
# mask = np.zeros(thresh.shape, dtype="uint8")

# # loop over the unique components
# for label in np.unique(labels):
# 	# if this is the background label, ignore it
# 	if label == 0:
# 		continue

# 	# otherwise, construct the label mask and count the
# 	# number of pixels 
# 	labelMask = np.zeros(thresh.shape, dtype="uint8")
# 	labelMask[labels == label] = 255
# 	numPixels = cv.countNonZero(labelMask)

# 	# if the number of pixels in the component is sufficiently
# 	# large, then add it to our mask of "large blobs"
# 	if numPixels > 300:
# 		mask = cv.add(mask, labelMask)

# # perform a connected component analysis on the thresholded
# # image, then initialize a mask to store only the "large"
# # components
# labels = measure.label(thresh, neighbors=8, background=0)
# mask = np.zeros(thresh.shape, dtype="uint8")
 
# # loop over the unique components
# for label in np.unique(labels):
# 	# if this is the background label, ignore it
# 	if label == 0:
# 		continue
 
# 	# otherwise, construct the label mask and count the
# 	# number of pixels 
# 	labelMask = np.zeros(thresh.shape, dtype="uint8")
# 	labelMask[labels == label] = 255
# 	numPixels = cv.countNonZero(labelMask)
 
# 	# if the number of pixels in the component is sufficiently
# 	# large, then add it to our mask of "large blobs"
# 	if numPixels > 300:
# 		mask = cv.add(mask, labelMask)




 
# # show the output imgage
# plt.imshow(mask)
# plt.show()
# pdb.set_trace()






# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# # sure background area
# sure_bg = cv.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg,sure_fg)

# # Marker labelling
# ret, markers = cv.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0


# markers = cv.watershed(src,markers)
# src[markers == -1] = [255,0,0]



# # You need to choose 4 or 8 for connectivity type
# connectivity = 4
# # Perform the operation
# output = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
# # Get the results
# # The first cell is the number of labels
# num_labels = output[0]
# # The second cell is the label matrix
# labels = output[1]
# # The third cell is the stat matrix
# stats = output[2]
# # The fourth cell is the centroid matrix
# centroids = output[3]


# # # color = (0,255,0)

# a = cv.imread('dilation/4050.jpg')

# for i, xy in enumerate(centroids):
#     x, y = int(xy[0]), int(xy[1])
#     # print(x,y)
#     cv.circle(a, (x, y), 10, (255, 0, 0),2)

# plt.figure()
# plt.imshow(a)
# plt.show()

# pdb.set_trace()