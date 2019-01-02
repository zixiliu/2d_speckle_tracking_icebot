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
    coordinates = peak_local_max(imgray, threshold_abs = 100, footprint = np.ones((5, 5)))
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

def get_color(x1,y1,x2,y2):

    intensity = 255

    yellow = (intensity,intensity,0)
    blue= (0,0,intensity)
    green = (0,intensity,0)
    light_blue=(0,intensity,intensity)

    if (x2-x1)>= 0 and (y2-y1)>=0:
        return yellow
    if (x2-x1)<= 0 and (y2-y1)>=0:
        return blue
    if (x2-x1)<= 0 and (y2-y1)<=0:
        return green
    if (x2-x1)>= 0 and (y2-y1)<=0:
        return light_blue
    # length = helper_get_distance(x1, y1, x2, y2)
    # if length > 0: 
    #     norm_x = ((x2-x1)/length+1)*127
    #     norm_y = ((y2-y1)/length+1)*127
    #     try: 
    #         color = (int(norm_x), 255, int(norm_y)) 
    #     except: 
    #         pdb.set_trace()
    #     return color
    # else: 
    #     return (0,255,0)


def find_match(prev_coordinates, new_coordinates, prev_img, new_img, method=cv.TM_CCOEFF, size = 7):
    '''size: half block size''' 
    intensity = 255

    yellow = (intensity,intensity,0)
    blue= (0,0,intensity)
    green = (0,intensity,0)
    light_blue=(0,intensity,intensity)
    red = (intensity,0,0)

    for new_xy in new_coordinates:
        new_y, new_x  = new_xy[0], new_xy[1]

        is_edge = helper_is_edge(new_x, new_y, new_img.shape, size)
        if is_edge == False:
            ## Find potential match within 30 pixels
            potential_match = []
            for j, prev_xy in enumerate(prev_coordinates):  
                prev_y, prev_x  = prev_xy[0], prev_xy[1]
                is_edge = helper_is_edge(prev_x, prev_y, prev_img.shape, size)
                if (helper_get_distance(new_x, new_y, prev_x, prev_y) <= 15) and (is_edge == False): 
                    potential_match.append([prev_xy, j])
            
            if len(potential_match) > 0:
                ## Block Match to find best match
                match_results = []
                for entry in potential_match: 
                    pt, j = entry[0], entry[1]
                    y, x = pt[0], pt[1]
                    res = cv.matchTemplate(prev_img[x-size:x+size+1, y-size:y+size+1],
                                new_img[new_x-size:new_x+size+1, new_y-size:new_y+size+1],method)                    
                    match_results.append([res[0][0], x, y, j])
                match_results = np.array(match_results)
                
                if len(match_results) > 0:
                    match_idx = np.argmax(match_results, axis=0)[0]
                    match = match_results[match_idx]
                    match_val, match_x, match_y, j= match[0], int(match[1]), int(match[2]), int(match[3])
                    if match_val > 10000:   

                        ## remove this maxima from the previous coordinates to achive 1 to 1 mapping
                        prev_coordinates = np.delete(prev_coordinates, j, axis=0) 
                        cv.circle(prev_img, (match_x, match_y), 2, red)
                        color = get_color(match_x, match_y, new_x, new_y)
                        cv.arrowedLine(new_img, (match_x, match_y), (new_x, new_y), color, 1, tipLength=0.3)

                        # line_length = helper_get_distance(match_x, match_y, new_x, new_y)
                        # if (line_length < 5): 
                        #     cv.circle(new_img, (new_x, new_y), 4, yellow, cv.FILLED)
                        # elif (line_length < 10): 
                        #     cv.circle(new_img, (new_x, new_y), 4, blue, cv.FILLED)
                        # elif (line_length < 15): 
                        #     cv.circle(new_img, (new_x, new_y), 4, green, cv.FILLED)
                        # else: 
                        #     cv.circle(new_img, (new_x, new_y), 4, light_blue, cv.FILLED)

                # cv.circle(new_img, (new_x, new_y), 2, (255, 0, 0))
                new_img[new_y, new_x] = np.array([255,0,0])


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

