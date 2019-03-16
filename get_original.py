import cv2
import numpy as np
import sys
import pdb
import glob
import matplotlib.pyplot as plt

### This script removes the other information in the original image and only leave the echo

def get_ori(filename):
    img = cv2.imread(filename)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(imgray,5,255,0)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)

    ori = img.copy()
    img = cv2.fillPoly(img, pts =[c], color=(255,255,255))
    fname = filename[-8::]
    plt.imsave('images/original/'+fname, ori - img)

file_path = "images/exp5_images/"
files = glob.glob(file_path+"*.jpg")
files = sorted(files)
file_of_interest = files[0:35]

for i in file_of_interest:
    print(i)
    get_ori(i)