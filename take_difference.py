import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.misc



from sift import helper_strip_img_text, helper_find_matches
import sys
import pdb
import glob


mypath='images/dilation/dilation10/'
# file1 = mypath+"16062017_125453973_4050.jpg"
# # # file2 = mypath+'16062017_125454009_4051.jpg'
# # # file2 = mypath+'16062017_125454209_4057.jpg'
# # # file2 = mypath + '16062017_125454573_4068.jpg'
# file2 = mypath +'16062017_125456445_4124.jpg'


# im1 = helper_strip_img_text(file1) 
# im2 = helper_strip_img_text(file2) 
# im = im2 - im1 

files = glob.glob(mypath+"*.jpg")
files = sorted(files)
start_idx = 0
file_of_interest = files[start_idx:start_idx+19]
for i in range(0, len(file_of_interest)-1):
    this_file = file_of_interest[i]
    print(this_file)
    next_file = file_of_interest[i+1]
    im1 = helper_strip_img_text(this_file) 
    im2 = helper_strip_img_text(next_file)  

    im = im2 - im1
    # idx = (im > 100) | (im < 50)
    # idx = (im > 200) | (im < 50)
    idx = (im>200)
    im[idx] = 0
    # pdb.set_trace()
    # Normalize
    # im = im/100*255

    # pdb.set_trace()
    plt.figure()
    plt.imshow(im)
    plt.imsave('images/diff_dilation/diff_dilation10/'+this_file[-8::], im)



## 3D plot
# # downscaling has a "smoothing" effect
# lena = scipy.misc.imresize(im, 0.15, interp='cubic')

# # create the x and y coordinate arrays (here we just use pixel indices)
# xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

# # create the figure
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.gray,
#         linewidth=0)

# # show it
# plt.show()


