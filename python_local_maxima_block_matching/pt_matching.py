import cv2 as cv
import matplotlib.pyplot as plt
# from scipy import ndimage
import numpy as np
# import sys
import pdb
import glob
# import colorsys
from skimage.feature import peak_local_max

from remove_outlier import remove_outlier


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

def helper_get_potential_match(new_x, new_y, prev_coordinates, prev_img, new_img, size, method):
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

def helper_get_best_match(match_results, new_x, new_y): 
    # match_idx = np.argmax(match_results, axis=0)[0]
    sort_idx = match_results.argsort(axis=0)[::-1][:,0]
    match = match_results[sort_idx[0]]

    # sorted_match_results = np.sort(match_results, axis=0)[::-1]
    # match = sorted_match_results[0]
    # pdb.set_trace()
    
    if match[0] > 5000: 
    # if match[0] > 0.90: 
        try:
            pt2 = match_results[sort_idx[1]]
            val1, val2 = match[0], pt2[0]
            # print(val2/val1)
            if sum(match_results[:,0]/val1 == 1) > 1: 
                if 1 in match_results[:,0]/val1:
                    print(val1, match_results[:,0]/val1)
                    # pdb.set_trace()
                    print('num matches: ', sum(match_results[:,0]/val1 == 1) -1)
            # print('best match', match[0], 'second best match', match_results[np.argmax(match_results, axis=0)[1]][0])
            if (val2/val1 > 0.95): 
                # print('two similar matches')
                match1_x, match1_y = match[1], match[2]
                match2_x, match2_y = pt2[1], pt2[2]
                dis1 = helper_get_distance(new_x, new_y, match1_x, match1_y)
                dis2 = helper_get_distance(new_x, new_y, match2_x, match2_y)
                if dis1 > dis2: 
                    match=pt2
                    # print((match1_x, match1_y), (match2_x, match2_y))
                # elif dis1==dis2:
                #     print('Two points of same distance and same val', dis1, val1)
                #     print((match1_x, match1_y), (match2_x, match2_y))
                #     pdb.set_trace()

        except: 
            # print("only one match")
            pass
    
        return match
    else: 
        return None
    
def helper_find_best_match(new_x, new_y, match_dict, prev_coordinates, size, prev_img, new_img, method):
    match_results = helper_get_potential_match(new_x, new_y, prev_coordinates, prev_img, new_img, size, method)
    while len(match_results) > 0:
        match = helper_get_best_match(match_results, new_x, new_y)
        try: 
            if len(match) >1: 
                match_val, match_x, match_y, j= match[0], int(match[1]), int(match[2]), int(match[3])

                this_match = (match_x, match_y)
                if this_match in match_dict: 
                    prev_match_newframe = match_dict[this_match]
                    prev_match_x, prev_match_y = prev_match_newframe[0], prev_match_newframe[1]
                    prev_result = cv.matchTemplate(prev_img[match_x-size:match_x+size+1, match_y-size:match_y+size+1],
                        new_img[prev_match_x-size:prev_match_x+size+1, prev_match_y-size:prev_match_y+size+1],method)[0][0]
                    if prev_result >= match_val: ## Previous match was a better match
                        match_results = np.delete(match_results, 0, axis=0)
                        
                    else: ## This match is a better match
                        match_dict[this_match] = (new_x, new_y)
                        match_dict = helper_find_best_match(prev_match_x, prev_match_y, match_dict, prev_coordinates, size, prev_img, new_img, method)
                        break
                else: 
                    match_dict[this_match] = (new_x, new_y)
                    break
            else: 
                break
        except: 
            break

    # print('hi')
    # pdb.set_trace()
    return match_dict

def helper_get_color(x1,y1,x2,y2):

    intensity = 255

    yellow = (intensity,intensity,0)
    blue= (0,0,intensity)
    green = (0,intensity,0)
    light_blue=(0,intensity,intensity)

    if (x2-x1)>= 0 and (y2-y1)>=0:
        return yellow
    if (x2-x1)<= 0 and (y2-y1)<=0:
        return green
    if (x2-x1)<= 0 and (y2-y1)>=0:
        return blue
    if (x2-x1)>= 0 and (y2-y1)<=0:
        return light_blue

################################################################################
########## Private Functions
################################################################################
def priv_get_local_max(imgray): 
    if len(imgray.shape)> 2:
        print("Expecting single channel gray image input to priv_get_local_max()")
        raise ValueError

    fp = np.ones((5, 5))
    fp[:,2] = np.zeros(5)
    fp[2,2] = 1

    coordinates = peak_local_max(imgray, threshold_abs = 100, footprint = fp)
    peaks = []
    for xy in coordinates: 
        x, y = xy[0], xy[1]
        peaks.append(imgray[x, y])
    sort_index = np.argsort(peaks)
    sort_index = sort_index[::-1]
    
    coordinates = coordinates[sort_index]
    return coordinates

'''Returns a 1:1 matched coordinate list'''
def priv_pt_matching(prev_coordinates, new_coordinates, prev_img, new_img, method=cv.TM_CCORR, size = 7):
    # pdb.set_trace()

    ## Dictionary of local maxima matches
    match_dict = {} # key: value ==> prev_pt: new_pt
    for i, new_xy in enumerate(new_coordinates):
        new_y, new_x  = new_xy[0], new_xy[1]
        is_edge = helper_is_edge(new_x, new_y, new_img.shape, size)
        if is_edge == False:
            match_dict = helper_find_best_match(new_x, new_y, match_dict, prev_coordinates, size, prev_img, new_img, method)
    ## Convert match dictionary to a list
    # match_list = [[x,match_dict[x]] for x in match_dict.keys()]
    return match_dict


def priv_draw_displacement(match_list, prev_file, new_file, new_img):

    prev_img = cv.imread(prev_file)
    prev_img_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    prev_coordinates = priv_get_local_max(prev_img_gray)
    
    if np.count_nonzero(new_img) == 0: 
        new_img = cv.imread(new_file)
        new_img_gray = cv.cvtColor(new_img,cv.COLOR_BGR2GRAY)
    else: 
        print("valid input new_img")
        new_img_gray = cv.cvtColor(new_img,cv.COLOR_BGR2GRAY)

    new_coordinates = priv_get_local_max(new_img_gray)


    for match in match_list: 
        prev_xy, new_xy = match[0], match[1]
        new_x, new_y = new_xy[0], new_xy[1]
        match_x, match_y = prev_xy[0], prev_xy[1]

        color = helper_get_color(match_x, match_y, new_x, new_y)
        cv.arrowedLine(new_img, (match_x, match_y), (new_x, new_y), color, 1, tipLength=0.3)
        # cv.circle(new_img, (new_x, new_y), 2, (255, 0, 0))

    for new_xy in new_coordinates:
        new_y, new_x  = new_xy[0], new_xy[1]    
        new_img[new_y, new_x] = np.array([255,0,0])
    for prev_xy in prev_coordinates: 
        prev_y, prev_x  = prev_xy[0], prev_xy[1]
        cv.circle(prev_img, (prev_x, prev_y), 2, (255, 0, 0))
    
    # fig = plt.figure(figsize=(12,9),frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(prev_img)
    # plt.savefig('results/'+prev_file[-8::])
    
    # fig = plt.figure(figsize=(16,12),frameon=False)
    fig = plt.figure(figsize=(20,20),frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(new_img)
    # plt.show()
    plt.savefig('images/3hierarchy_ccorr_normed/'+new_file[-8::]) 
    # plt.figure(figsize=(20,20))   
    # plt.imsave('images/3hierarchy_ccorr_normed/'+new_file[-8::], new_img)


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

    match_dict = priv_pt_matching(prev_coordinates, new_coordinates, prev_img, new_img)
    match_dict, new_img = remove_outlier(match_dict, new_img.shape, new_img, method='neighbor_pixel')
    # ## Second round
    # match_dict, new_img = remove_outlier(match_dict, new_img.shape, new_img, method='neighbor_pixel')
    # ## Third round
    # match_dict, new_img = remove_outlier(match_dict, new_img.shape, new_img, method='neighbor_pixel')
    # # ## Four round
    # match_dict, new_img = remove_outlier(match_dict, new_img.shape, new_img, method='neighbor_pixel')
    # ## Fifth round
    # match_dict, new_img = remove_outlier(match_dict, new_img.shape, new_img, method='neighbor_pixel')
    # ## Sixth round
    # match_dict, new_img = remove_outlier(match_dict, new_img.shape, new_img, method='neighbor_pixel')
    
    # # Convert match dictionary to a list
    match_list = [[x,match_dict[x]] for x in match_dict.keys()]
    # priv_draw_displacement(match_list, prev_file, new_file, np.zeros(new_img.shape))
    # match_list = None
    priv_draw_displacement(match_list, prev_file, new_file, new_img)


    # pdb.set_trace()


################################################################################
## Delete me
################################################################################
# ori_path = '../original/'
# # ori_files = glob.glob(ori_path+"*.jpg")
# # ori_files = sorted(ori_files)
# file1 = ori_path+"4056.jpg"
# file2 = ori_path+'4057.jpg'

# match_frames(file1, file2)


ori_path = '../original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

for i, file in enumerate(ori_files):
    if i < len(ori_files)-1:
        print(file)
        print(ori_files[i+1])
        match_frames(file, ori_files[i+1])