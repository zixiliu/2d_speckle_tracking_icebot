import cv2 as cv
# from scipy import ndimage
import numpy as np
# import sys
# import pdb
# import glob
# import colorsys
from skimage.feature import peak_local_max

################################################################################
########## Helper Functions
################################################################################
def helper_is_edge(x, y, imshape, size):
    row, col = imshape[:2] 
    if (x-size<0) or (x+size>row) or (y-size<0) or (y+size>col): 
        return True
    else: 
        return False

def helper_get_distance(x1, y1, x2, y2):
	'''Get distance between two keypoints'''
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def helper_get_potential_match(new_x, new_y, prev_coordinates, prev_img, new_img, size):
    potential_match = []
    for j, prev_xy in enumerate(prev_coordinates):  
        prev_y, prev_x  = prev_xy[0], prev_xy[1]
        is_edge = helper_is_edge(prev_x, prev_y, prev_img.shape, size)
        if (helper_get_distance(new_x, new_y, prev_x, prev_y) <= 15) and (is_edge == False): 
            potential_match.append([prev_xy, j])

    ## Block Match to find best match
    match_results = []
    for entry in potential_match: 
        pt, j = entry[0], entry[1]
        y, x = pt[0], pt[1]
        res = cv.matchTemplate(prev_img[x-size:x+size+1, y-size:y+size+1],
                    new_img[new_x-size:new_x+size+1, new_y-size:new_y+size+1],method)                    
        match_results.append([res[0][0], x, y, j])
    match_results = np.array(match_results)
    return match_results

def helper_get_best_match(match_results): 
    match_idx = np.argmax(match_results, axis=0)[0]
    match = match_results[match_idx]

    if match[0] > 10000: 
        return match
    else: 
        return None
    
def helper_find_best_match(new_x, new_y, match_dict, prev_coordinates, size, prev_img, new_img, method):
    match_results = helper_get_potential_match(new_x, new_y, prev_coordinates, prev_img, new_img, size)
    while len(match_results) > 0:
        match = helper_get_best_match(match_results)
        if match != None: 
            match_val, match_x, match_y, j= match[0], int(match[1]), int(match[2]), int(match[3])

            this_match = (match_x, match_y)
            if this_match in match_dict: 
                prev_match_newframe = match_dict[this_match]
                prev_match_x, prev_match_y = prev_match_newframe[0], prev_match_newframe[1]
                prev_result = cv.matchTemplate(prev_img[match_x-size:match_x+size+1, match_y-size:match_y+size+1],
                    new_img[prev_match_x-size:prev_match_x+size+1, prev_match_y-size:prev_match_y+size+1],method)
                if prev_result >= match_val: ## Previous match was a better match
                    np.delete(match_results, 0, axis=0)
                else: ## This match is a better match
                    match_dict[this_match] = (new_x, new_y)
                    match_dict = helper_find_best_match(prev_match_x, prev_match_y, match_dict, prev_coordinates, size, prev_img, new_img, method)
    return match_dict

################################################################################
########## Private Functions
################################################################################
def priv_get_local_max(imgray): 
    if len(imgray.shape)> 2:
        print("Expecting single channel gray image input to priv_get_local_max()")
        raise ValueError

    coordinates = peak_local_max(imgray, threshold_abs = 100, footprint = np.ones((5, 5)))
    peaks = []
    for xy in coordinates: 
        x, y = xy[0], xy[1]
        peaks.append(imgray[x, y])
    sort_index = np.argsort(peaks)
    sort_index = sort_index[::-1]
    
    coordinates = coordinates[sort_index]
    return coordinates

'''Returns a 1:1 matched coordinate list'''
def priv_pt_matching(prev_coordinates, new_coordinates, prev_img, new_img, method=cv.TM_CCOEFF, size = 7):
    ## Dictionary of local maxima matches
    match_dict = {} # key: value ==> prev_pt: new_pt
    for new_xy in new_coordinates:
        new_y, new_x  = new_xy[0], new_xy[1]
        is_edge = helper_is_edge(new_x, new_y, new_img.shape, size)
        if is_edge == False:
            match_dict = helper_find_best_match(new_x, new_y, match_dict, prev_coordinates, size, prev_img, new_img, method)
            
    ## Convert match dictionary to a list
    match_list = [[x,match_dict[x]] for x in match_dict.keys()]
    return match_list

################################################################################
########## Public Functions
################################################################################
def match_frames(prev_file, new_file):
    prev_img = cv.imread(prev_file)
    prev_img_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    prev_coordinates = priv_get_local_max(prev_img_gray)

    new_img = cv.imread(new_file)
    new_img_gray = cv.cvtColor(new_img,cv.COLOR_BGR2GRAY)
    new_coordinates = priv_get_local_max(new_img_gray)

    match_list = priv_pt_matching(prev_coordinates, new_coordinates, prev_img, new_img)
