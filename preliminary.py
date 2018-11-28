import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb
import glob
import os
### My functions
from speckle import find_blob


mypath = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/"



# filename = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/16062017_125258857_600.jpg"
# find_blob(filename)

# filename = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/16062017_125258889_601.jpg"
# find_blob(filename)

# filename = "/Volumes/GoogleDrive/My Drive/Alperen's Images/Exp 5 The Best Set/20170616_124815718/images/16062017_125258925_602.jpg"
# find_blob(filename)



print(mypath+"*.jpg")
files = glob.glob(mypath+"*.jpg")

file_of_interest = files[-10::]

for f in file_of_interest: 
	find_blob(f)