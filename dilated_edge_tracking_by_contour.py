'''This file automatically tracks the edge of the valve with inputs of thresholded dilated images, 
    saves the results in both image and numpy array format'''

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


def get_edge_pts(mypath): 

    ori_path = 'images/original/'
    ori_files = glob.glob(ori_path+"*.jpg")
    ori_files = sorted(ori_files)

    files = glob.glob(mypath+"*.jpg")
    files = sorted(files)
    start_idx = 0
    file_of_interest = files[start_idx:start_idx+20]

    num_pts = 20

    pts = np.zeros((20, num_pts, 2))

    for file_idx in range(0, len(file_of_interest)):
        # print(file_idx)
        this_file = file_of_interest[file_idx]
        ori_file = ori_files[file_idx]
        img = helper_strip_img_text(this_file) 
        ori_img = cv2.imread(ori_file)

        edges = cv2.Canny(img,100,200)
        row, col = edges.shape
        th = 150
        for i in range(row):
            for j in range(col): 
                if (i < row/2 - th) or (i > row/2 + th) or (j < col/2 - th) or (j > col/2 + th):
                    edges[i][j] = 0

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = edges.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # Floodfill from point (0, 0)
        cv2.floodFill(edges, mask, (0,0), 255)
        
        # Invert floodfilled image
        edges_inv = cv2.bitwise_not(edges)

        edges_inv = cv2.dilate(edges_inv, None, iterations=20)

        _, contours, _ = cv2.findContours(edges_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)
        cv2.drawContours(ori_img, contours, 0, (255, 0, 0), 1)
        try:
            ctr_index = np.linspace(1,len(contours[0]),num_pts+1)
            ctr_index = ctr_index[0:-1]
            # pdb.set_trace()
        except:
            print('contour problem')
            pdb.set_trace()
        for j, idx in enumerate(ctr_index): 
            idx = (int(idx)-1)
            xy = contours[0][idx][0]
            if j == 0: 
                cv2.circle(ori_img, (xy[0], xy[1]), 5, (255,0,0))    
            cv2.circle(ori_img, (xy[0], xy[1]), 3, (0,255,0))
            pts[file_idx][j] = xy
            ## draw arrows:
            if file_idx > 0: 
                match_x, match_y = int(pts[file_idx-1][j][0]), int(pts[file_idx-1][j][1])
                cv2.arrowedLine(ori_img, (match_x, match_y), (xy[0], xy[1]), (0,100,0), 1, tipLength=0.3)

        plt.imsave('images/dilation_edge_tracking/'+this_file[-8::], ori_img)
    
    return pts

pts= get_edge_pts(mypath)
print(pts.shape)
np.save('outputs/auto_pts', pts)