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


def mark_blob(filename):

	image = cv.imread(filename)
	image_gray = rgb2gray(image)

	blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
	pdb.set_trace()
	# Compute radii in the 3rd column.
	blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

	fig, ax = plt.subplots()

	plt.imshow(image, interpolation='nearest')

	for blob in blobs_log:
	    y, x, r = blob
	    c = plt.Circle((x, y), r, color='lime', linewidth=2, fill=False)
	    ax.add_patch(c)

	plt.tight_layout()
	# plt.show()
	plt.savefig("laplacian_blob/"+filename[-8::])


ori_path = 'exp5_images/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

for file in ori_files: 
	print(file)
	mark_blob(file)

