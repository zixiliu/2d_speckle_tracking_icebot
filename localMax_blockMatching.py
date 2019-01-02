import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb
import glob
import colorsys
from skimage.feature import peak_local_max

ori_path = 'original/'
# ori_files = glob.glob(ori_path+"*.jpg")
# ori_files = sorted(ori_files)
file1 = ori_path+"4056.jpg"
file2 = ori_path+'4057.jpg'

def helper_get_distance(x1, y1, x2, y2):
	'''Get distance between two keypoints'''
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_local_max(imgray): 
    coordinates = peak_local_max(imgray, footprint = np.ones((12, 12)))
    peaks = []
    for xy in coordinates: 
        x, y = xy[0], xy[1]
        peaks.append(imgray[x, y])
    sort_index = np.argsort(peaks)
    sort_index = sort_index[::-1]
    
    coordinates = coordinates[sort_index]
    # peaks = np.array(peaks)[sort_index]

    return coordinates

def plot_local_max(filename): 
    ori = cv.imread(filename)
    ori_gray = cv.cvtColor(ori,cv.COLOR_BGR2GRAY)
    coordinates = get_local_max(ori_gray)

    for i, item in enumerate(coordinates): 
        y, x  = item[0], item[1]
        cv.circle(ori, (x, y), 1, (255, 0, 0))
        if i in range(400, 450):
            top_left = (x-7, y+7)
            bottom_right = (x+7, y-7)
            cv.rectangle(ori, top_left, bottom_right, (0,255, 0))

    plt.figure(figsize=(30,25))
    plt.imshow(ori)
    plt.savefig("localmax_blockmatch/"+filename[-8::])

def helper_is_edge(x, y, imshape, size):
    row, col = imshape[:2] 
    if (x-size<0) or (x+size>row) or (y-size<0) or (y+size>col): 
        return True
    else: 
        return False


def find_match(prev_coordinates, new_coordinates, prev_img, new_img, method=cv.TM_CCOEFF, size = 5):
    '''size: half block size''' 
    for i, new_xy in enumerate(new_coordinates):
        new_y, new_x  = new_xy[0], new_xy[1]
        
        def check_dis(pt): 
            y, x = pt[0], pt[1]
            return helper_get_distance(new_x, new_y, x, y)



        is_edge = helper_is_edge(new_x, new_y, new_img.shape, size)
        if is_edge == False:
            ## Find potential match within 30 pixels
            potential_match = []
            for j, prev_xy in enumerate(prev_coordinates):  
                prev_y, prev_x  = prev_xy[0], prev_xy[1]
                is_edge = helper_is_edge(prev_x, prev_y, prev_img.shape, size)
                if (helper_get_distance(new_x, new_y, prev_x, prev_y) <= 15) and (is_edge == False): 
                    potential_match.append(prev_xy)
            
            if len(potential_match) > 0:
                ## Block Match to find best match
                match_results = []
                for pt in potential_match: 
                    y, x = pt[0], pt[1]
                    res = cv.matchTemplate(prev_img[x-size:x+size, y-size:y+size],
                                new_img[new_x-size:new_x+size, new_y-size:new_y+size],method)
                    if res >= 50000:
                        # match_results.append(res[0][0])
                        match_results.append(pt)                
                # match_results = np.array(match_results)
                if len(match_results) > 0:
                    match_sorted = sorted(match_results, key=check_dis)


                    # sort_index = np.argsort(match_results)
                    # sort_index = sort_index[::-1]
                    # match_results = match_results[sort_index]
                    

                    # match = potential_match[match_idx]
                    match = match_sorted[0]
                    np.delete(prev_coordinates, j)
                    ## Draw a square around match
                    if i <= 1000:
                        # top_left = (new_x-size, new_y+size)
                        # bottom_right = (new_x+size, new_y-size)
                        # cv.rectangle(new_img, top_left, bottom_right, (0,255, 0))

                        match_y, match_x = match[0], match[1]
                        cv.circle(prev_img, (match_x, match_y), 2, (255, 0, 0))
                        # top_left = (match_x-size, match_y+size)
                        # bottom_right = (match_x+size, match_y-size)
                        # cv.rectangle(prev_img, top_left, bottom_right, (255,255, 0))

                        cv.arrowedLine(new_img, (match_x, match_y), (new_x, new_y), (255, 0, 0), 1, tipLength=0.3)
                # cv.circle(new_img, (new_x, new_y), 2, (0, 255, 0))
                new_img[new_y, new_x] = np.array([0,255,0])


    return new_img, prev_img



def main(file1, file2): 
    
    im1 = cv.imread(file1)
    im1_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
    prev_coordinates = get_local_max(im1_gray)

    im2 = cv.imread(file2)
    im2_gray = cv.cvtColor(im2,cv.COLOR_BGR2GRAY)
    new_coordinates = get_local_max(im2_gray)

    # pdb.set_trace()
    new_img, prev_img = find_match(prev_coordinates, new_coordinates, im1, im2)

    fig = plt.figure(figsize=(64,48),frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(prev_img)
    plt.savefig("localmax_blockmatch/"+file1[-8::])
    
    fig = plt.figure(figsize=(64,48),frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(new_img)
    plt.savefig("localmax_blockmatch/"+file2[-8::])
    

main(file1, file2)
# plot_local_max(file2)

