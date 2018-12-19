import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb
import glob


# Read the image you want connected components of
src = cv2.imread('dilation/4050.jpg')
# Threshold it so it becomes binary

imgray = src.copy()
imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
imgray_cp = imgray.copy()
thresh = imgray_cp
# print(imgray_cp.shape)
# print(imgray_cp[:,:,0].max(), imgray_cp[:,:,1].max(), imgray_cp[:,:,2].max())
# ret,thresh = cv2.threshold(imgray_cp,255,2555,0)


# plt.imshow(thresh)
# plt.show()

# ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# You need to choose 4 or 8 for connectivity type
connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]



pdb.set_trace()