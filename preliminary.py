import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb
import glob
import os
### My functions
from speckle import find_blob, two_frames, process_frames
import sift

# mypath = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/"
# filename = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/16062017_125258857_600.jpg"
# filename = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/16062017_125258889_601.jpg"


mypath = "exp5_images/"
file1 = mypath+"16062017_125258857_600.jpg"
file2 = mypath+'16062017_125258889_601.jpg'


# print(mypath+"*.jpg")
# files = glob.glob(mypath+"*.jpg")

# file_of_interest = files[-10::]

# for f in file_of_interest: 
# 	find_blob(f)



# two_frames(file1, file2)
# 
process_frames(mypath)
# diff_path = "diff/"
# process_frames(diff_path)

dilation_path = 'dilation/'
sift.process_consecutive_frames(dilation_path)
# sift.process_consecutive_frames(mypath)
# sift.process_same_frames(mypath)

# diff_path = "diff/"
# sift.process_consecutive_frames(diff_path)
