import cv2 as cv
import pdb
import glob

import numpy as np
# from skimage.feature import peak_local_max
from helper_get_pts import manual_select_and_save, helper_record_speckles_in_one_frame

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from python_hierachy_block_matching.block_matching import block_match
from python_hierachy_block_matching.trace_trajectory import find_match
import os

################################################################################
## Global variables
################################################################################
# ori_path = '/Users/zixiliu/my_git_repos/my_howe_lab/Echos/Zeo file/IM_1637_copy_jpg/'
ori_path = '/Users/zixiliu/Documents/Daichi/20190926_Images/20190621104753_S8_probe/reordered/'
save_img_path = ori_path+'tracking/'
try:
    os.mkdir(save_img_path)
except OSError:
    pass

manually_select_new_pts = False
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)
blur_window_size = 9
starting_img_num = 1
ending_img_num = 34

################################################################################
## Helper functions
################################################################################
'''Given (x, y), find match in previous frame.'''
def helper_find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method):
    global blur_window_size
    ## Gaussian Blur
    prev_gaussian = cv.GaussianBlur(prev_gray,(blur_window_size,blur_window_size),0)
    next_gaussian = cv.GaussianBlur(next_gray,(blur_window_size,blur_window_size),0)

    ## Define template by boundary points
    template_yy_ul, template_yy_lr = max(0,y-half_template_size), min(prev_gray.shape[0],y+half_template_size+1)
    template_xx_ul, template_xx_lr = max(0,x-half_template_size), min(prev_gray.shape[1],x+half_template_size+1)
    template = prev_gaussian[template_yy_ul:template_yy_lr, template_xx_ul:template_xx_lr]
    ## define source around the previous match
    yy_ul, yy_lr = y-half_source_size, y+half_source_size+1
    xx_ul, xx_lr =  x-half_source_size, x+half_source_size+1
    if yy_ul < 0:
        yy_ul = 0
    if xx_ul < 0:
        xx_ul = 0
    if yy_lr > prev_gray.shape[0]-1:
        yy_lr = prev_gray.shape[0]-1
    if xx_lr > prev_gray.shape[1]-1:
        xx_lr = prev_gray.shape[1]-1
    source = next_gaussian[yy_ul:yy_lr, xx_ul:xx_lr]
    ## Block match
    (next_match_x, next_match_y) = block_match(source, template, method, half_template_size, \
                                    yy_ul, yy_lr, xx_ul, xx_lr, x,y,\
                                    template_yy_ul, template_yy_lr, template_xx_ul, template_xx_lr)
    if next_match_x == -1:
        (next_match_x, next_match_y) = x, y
    if next_match_x == np.Inf:
        return np.Inf, np.Inf
    return next_match_x, next_match_y

def helper_get_tracking_points(f):

    # coordinates = helper_record_speckles_in_one_frame(f)
    save_file = save_img_path+'manually_selected_pts.npy'
    if manually_select_new_pts:
        manual_select_and_save(f, save_file)
    coordinates = np.load(save_file)
    neighbors = {}

    # import matplotlib.pyplot as plt
    from python_hierachy_block_matching.block_matching import block_match
    from python_hierachy_block_matching.trace_trajectory import find_match
    return coordinates, neighbors

################################################################################
## Global functions
################################################################################
def gaussian_blur_bm(make_plots = False, half_template_size = 16, half_source_size = 23):
    global blur_window_size
    # xys, neighbors = helper_get_tracking_points(ori_path + str(starting_img_num) + '.jpg')
    xys, neighbors = helper_get_tracking_points(ori_files[0])
    method = cv.TM_CCORR_NORMED # Other methods: cv.TM_CCOEFF_NORMED #cv.TM_CCORR_NORMED

    num_pts = len(xys)
    for i, xy in enumerate(xys):
        x, y = xy[0], xy[1]
        xys[i] = [x, y]

    num_frames = ending_img_num - starting_img_num + 1
    xy_list = np.zeros((len(xys), num_frames, 2))
    xy_list[:,0,:] = xys
    center_pt = np.zeros((num_frames, 2))

    for ii, i in enumerate(range(starting_img_num,ending_img_num)):
        f1 = str(i)+'.jpg'
        f2 = str(i+1)+'.jpg'

        ori_file = ori_path + f2
        ori_prev = ori_path + f1
        # ori_file = ori_files[i]
        # ori_prev = ori_files[i+1]
        print(ori_prev)

        prev_img = cv.imread(ori_prev)
        prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
        next_img = cv.imread(ori_file)
        next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)

        if make_plots:
            plt.figure(figsize=(21, 12))
            plt.subplot(131)
            plt.imshow(next_img)
            plt.subplot(132)
            next_gaussian = cv.GaussianBlur(next_gray,(blur_window_size,blur_window_size),0)
            plt.imshow(next_gaussian, cmap='gray')
        elif make_plots:
            plt.figure(figsize=(15,15))

        ## Find match
        for j in range(num_pts):
            x, y = int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])
            next_match_x, next_match_y = helper_find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)

            if next_match_x == np.Inf:
                next_match_x, next_match_y = x, y

            xy_list[j,ii+1,:] = [next_match_x, next_match_y]
            if make_plots:
                cv.circle(next_img, (int(x), int(y)), 3, (0, 255, 0))
                cv.circle(next_img, (int(next_match_x), int(next_match_y)), 4, (255, 0, 0))
        if make_plots:
            plt.imshow(next_img)

            plt.tight_layout()
            plt.imsave(save_img_path+f2, next_img)
            plt.close()

        if ii == 0:
            for j in range(num_pts):
                x, y = xy_list[j,0,0], xy_list[j,0,1]
                cv.circle(prev_img, (int(x), int(y)), 4, (255, 0, 0))
            plt.imsave(save_img_path+f1, prev_img)

    np.save(save_img_path + 'tracked_pts.npy', xy_list)
    np.save(save_img_path + 'neighbors.npy', neighbors)
    np.save(save_img_path + 'center_pt.npy', center_pt)

if __name__ == "__main__":
    half_template_size = 30
    half_source_size = 38
    gaussian_blur_bm(half_template_size=half_template_size, half_source_size=half_source_size, make_plots=True)

