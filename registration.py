import cv2
import numpy as np
import matplotlib.pyplot as plt
from sift import helper_strip_img_text, helper_find_matches
import sys
import pdb
import glob


mypath = "exp5_images/"
file1 = mypath+"16062017_125453973_4050.jpg"
# file2 = mypath+'16062017_125454009_4051.jpg'
# file2 = mypath+'16062017_125454209_4057.jpg'
# file2 = mypath + '16062017_125454573_4068.jpg'
file2 = mypath +'16062017_125456445_4124.jpg'

def reg(file1, file2):


    # Read the images to be aligned
    # im1 =  cv2.imread(file1);
    # im2 =  cv2.imread(file2);
    im1 = helper_strip_img_text(file1) 
    im2 = helper_strip_img_text(file2) 
    
    this_img=im1
    next_img=im2

    this_kpts_wrt_next, next_kps, idx_train, idx_query_next = helper_find_matches(this_img, next_img, dis_thresh=10)

    # Extract location of good matches
    points1 = np.zeros((len(idx_query_next), 2), dtype=np.float32)
    points2 = np.zeros((len(idx_train), 2), dtype=np.float32)

    for i, idx in enumerate(idx_query_next):
        points1[i, :] = this_kpts_wrt_next[idx].pt
    for i, idx in enumerate(idx_train):
        points2[i, :] = next_kps[idx].pt
    
    # average_vector=np.mean(np.array(points1-points2),axis=0)
    # M = np.float32([[1,0,average_vector[0]],[0,1,average_vector[1]]])
    # height, width = im2.shape
    # im2Reg = cv2.warpAffine(im2,M,(width,height))

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = im2.shape
    im2Reg = cv2.warpPerspective(im2, h, (width, height))

    # super imposed
    super_impose = np.zeros([height, width, 3])
    super_impose[:,:,2] = im1/255
    super_impose[:,:,0] = im2Reg/255

    return im2Reg, super_impose, im1, im2


def plot_super(im1Reg, super_impose, im1, im2, file1, file2):
    # Show final results
    plt.figure()
    plt.subplot(321)
    plt.imshow(im1, cmap='gist_gray')
    plt.title(file1[-8::])
    plt.subplot(322)
    plt.title(file2[-8::])
    plt.imshow(im2, cmap='gist_gray')

    plt.subplot(323)
    plt.imshow(super_impose)
    plt.title('registered')

    plt.subplot(324)
    height, width = im2.shape
    super_impose = np.zeros([height, width, 3])
    super_impose[:,:,2] = im1/255
    super_impose[:,:,0] = im2/255
    plt.imshow(super_impose)
    plt.title('super imposed')

    plt.subplot(325)
    plt.imshow(im1Reg, cmap='gist_gray')
    plt.title("Translated "+file1[-8::])

    plt.tight_layout()
    plt.show()
    # plt.savefig('registration_same.png')

# def process_consecutive_frames(file_path):
#     files = glob.glob(file_path+"*.jpg")
#     files = sorted(files)
#     start_idx = 0
#     file_of_interest = files[start_idx:start_idx+19]
#     for i in range(0, len(file_of_interest)-1):
#         this_file = file_of_interest[i]
#         next_file = file_of_interest[i+1]
#         im1Reg, super_impose, im1, im2 = reg(this_file, next_file)
#         plot_super(im1Reg, super_impose, im1, im2, this_file, next_file )

# process_consecutive_frames(mypath)

im1Reg, super_impose, im1, im2 = reg(file1, file2)
plot_super(im1Reg, super_impose, im1, im2, file1, file2 )