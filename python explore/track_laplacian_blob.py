"""
==============
Blob Detection
==============

Blobs are bright on dark or dark on bright regions in an image. In
this example, blobs are detected using 3 algorithms. The image used
in this case is the Hubble eXtreme Deep Field. Each bright dot in the
image is a star or a galaxy.

Laplacian of Gaussian (LoG)
-----------------------------
This is the most accurate and slowest approach. It computes the Laplacian
of Gaussian images with successively increasing standard deviation and
stacks them up in a cube. Blobs are local maximas in this cube. Detecting
larger blobs is especially slower because of larger kernel sizes during
convolution. Only bright blobs on dark backgrounds are detected. See
:py:meth:`skimage.feature.blob_log` for usage.

Difference of Gaussian (DoG)
----------------------------
This is a faster approximation of LoG approach. In this case the image is
blurred with increasing standard deviations and the difference between
two successively blurred images are stacked up in a cube. This method
suffers from the same disadvantage as LoG approach for detecting larger
blobs. Blobs are again assumed to be bright on dark. See
:py:meth:`skimage.feature.blob_dog` for usage.

Determinant of Hessian (DoH)
----------------------------
This is the fastest approach. It detects blobs by finding maximas in the
matrix of the Determinant of Hessian of the image. The detection speed is
independent of the size of blobs as internally the implementation uses
box filters instead of convolutions. Bright on dark as well as dark on
bright blobs are detected. The downside is that small blobs (<3px) are not
detected accurately. See :py:meth:`skimage.feature.blob_doh` for usage.
"""

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
import cv2 as cv
import pdb
import glob
import numpy as np 
import colorsys

colors255 = [  (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
            (255, 255, 255), (127, 0, 0), (0, 127, 0), (0, 0, 127), (127, 127, 0),
            (0, 127, 127)]

colors = [  (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), 
            (1, 1, 1), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0),
            (0, 0.5, 0.5)]
N = len(colors)
# N = 70
# HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
# RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
# colors = RGB_tuples

num_colors = 0

class MyLaplacianBlob: 
	def __init__(self, x, y, r, color_index):
		self.x = x
		self.y = y
		self.r = r
		self.color_index = color_index

def mark_blob(filename):
	global num_colors, colors, N

	image = cv.imread(filename)
	image_gray = rgb2gray(image)

	blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
	# pdb.set_trace()	
	# Compute radii in the 3rd column.
	# blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

	fig, ax = plt.subplots()

	plt.imshow(image, interpolation='nearest')

	my_blobs = []
	for i, blob in enumerate(blobs_log):
	    y, x, r = blob
	    c = plt.Circle((x, y), r, color=colors[i % N], linewidth=1, fill=False)
	    ax.add_patch(c)

	    my_blob = MyLaplacianBlob(x, y, r, i)
	    my_blobs.append(my_blob)

	num_colors += len(blobs_log)
	plt.tight_layout()
	# plt.show()
	plt.savefig("laplacian_blob/"+filename[-8::])


	my_blobs = sorted(my_blobs, key=lambda x: x.r, reverse = True)
	return my_blobs

def helper_get_distance(x1, y1, x2, y2):
	'''Get distance between two keypoints'''
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def track_blob(filename, prev_blobs, ori_files): 
	global num_colors, colors, N

	if prev_blobs == None: 
		blobs_log = mark_blob(filename)
		return blobs_log
	else:
		### Find blob
		image = cv.imread(filename)
		image_gray = rgb2gray(image)
		ori_image = cv.imread(ori_files)
		
		blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
		# Compute radii in the 3rd column.
		blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
		
		blobs_log = sorted(blobs_log, key=lambda x: x[2], reverse = True)

		my_blobs = []
		for blob in blobs_log: 
			y, x, r = blob
			is_new_blob = False
			### Find potential matches of centroids within xx pixels of the current contour
			potential_match = []
			for prev_blob in prev_blobs: 
				if helper_get_distance(x, y, prev_blob.x, prev_blob.y) <= 20: 
					potential_match.append(prev_blob)
			### Find a match that has the most similar area
			if len(potential_match) > 0: 
				closest = min(potential_match, key=lambda x:abs(x.r-r))
				# closest = min(potential_match, key=lambda xx:helper_get_distance(x, y, xx.x, xx.y))                

				if ((0.4 * r < closest.r) & (closest.r < (4 * r))):
					this_blob = MyLaplacianBlob(x, y, r, closest.color_index)
					my_blobs.append(this_blob)
					
					prev_blobs.remove(closest)

					cv.arrowedLine(ori_image,(int(closest.x), int(closest.y)), (int(x), int(y)),colors255[closest.color_index % N], 2, tipLength=0.3)
				else: 
					is_new_blob = True
			else: 
				is_new_blob = True

			if is_new_blob:
				this_blob = MyLaplacianBlob(x, y, r, num_colors)
				num_colors += 1
				my_blobs.append(this_blob)
			
			c = plt.Circle((x, y), r, color=colors[this_blob.color_index % N], linewidth=1, fill=False)
			ax1.add_patch(c)
			
		ax1.imshow(ori_image, interpolation='nearest')
		ax2.imshow(ori_image, interpolation='nearest')
		plt.tight_layout()
		# plt.show()
		plt.savefig("laplacian_blob/"+filename[-8::])

		return my_blobs



dilation_path = 'original/'

dilation_files = glob.glob(dilation_path+"*.jpg")
dilation_files = sorted(dilation_files)



ori_path = 'original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)
# counter = 0
# while counter < 30: 
# 	for file in ori_files: 
# 		print(file)
# 		mark_blob(file)
# 		counter += 1

# The very first frame
i = 0
my_blobs = track_blob(ori_files[i], None, ori_files[i])
# The second frame forward
for i, file in enumerate(ori_files):
    my_blobs = track_blob(file, my_blobs, ori_files[i])
    print(file)
