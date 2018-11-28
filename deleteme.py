import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb

plotit = True
# ######################################
# ### Read In File
# ######################################
# filename = "dots.jpg"
# a = cv2.imread(filename)

# imgray = a.copy()
# imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)

# ######################################
# ### Mask Out Info by Contour 
# ######################################
# imgray_cp = imgray.copy()
# ret,thresh = cv2.threshold(imgray_cp,100,255,0)
# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=2)



# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # pdb.set_trace()

# if plotit:

# 	plt.figure(figsize=(15,15))

# 	plt.subplot(121)
# 	plt.imshow(thresh, cmap='gist_gray')
# 	plt.subplot(122)
# 	# plt.imshow(image,  cmap='gist_gray')
# 	cv2.drawContours(a, contours, -1, (0,255,0), 5)
# 	plt.imshow(a)
# 	# plt.title('Thresh after threshold imgray')
# 	plt.show()


# Read image
im = cv2.imread("dots.jpg", cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
 
print(im.shape)

# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# # Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()

# # Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200


# # Filter by Area.
# params.filterByArea = True
# params.minArea = 1500

# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1

# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87

# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

# # Create a detector with the parameters
# detector = cv2.SimpleBlobDetector_create(params)

# # Detect blobs.
# keypoints = detector.detect(im)

# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# # the size of the circle corresponds to the size of blob

# im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.imshow(im_with_keypoints)
plt.show()




