import cv2 as cv
from block_matching import block_match
from trace_trajectory import find_match
import pdb
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max


## Global variables 
path16 = '../images/downsamplePlot/down_by_16/'
path8 = '../images/downsamplePlot/down_by_8/'
path4 = '../images/downsamplePlot/down_by_4/'
path2 = '../images/downsamplePlot/down_by_2/'
ori_path = '../images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)
warp_x = np.load('warpx.npy')
warp_y = np.load('warpy.npy')

with_warp = True
blur_window_size = 9#5#49#5

# def smooth_outlier():


def find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method):

    global blur_window_size
    prev_gaussian = cv.GaussianBlur(prev_gray,(blur_window_size,blur_window_size),0)
    next_gaussian = cv.GaussianBlur(next_gray,(blur_window_size,blur_window_size),0)

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(prev_gray, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(prev_gaussian, cmap='gray')
    # plt.show()
    # pdb.set_trace()

    ## define template
    if  (y-half_template_size < 0) | (y+half_template_size+1 > prev_gray.shape[0]-1) | \
        (x-half_template_size < 0) | (x+half_template_size+1 > prev_gray.shape[1]-1):
        return np.Inf, np.Inf

    template = prev_gaussian[y-half_template_size:y+half_template_size+1, x-half_template_size:x+half_template_size+1]

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

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(template)
    # plt.subplot(122)
    # plt.imshow(source)
    # plt.show()
    # pdb.set_trace()

    (next_match_x, next_match_y) = block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y)
    return next_match_x, next_match_y

def get_tracking_points():
    global with_warp
    # '''Get points of interest based on intensity in downsampled by 8 images (for limited number to track)'''
    f = ori_path + '4051.jpg'
    img = cv.imread(f)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if with_warp: 
        gray_warp = cv.remap(gray, warp_x, warp_y, cv.INTER_LINEAR)
        img2 = cv.pyrDown(gray_warp)
    else:
        img2 = cv.pyrDown(gray)
    img4 = cv.pyrDown(img2)
    gray = cv.pyrDown(img4)

    _, thresh = cv.threshold(gray,80,1,cv.THRESH_BINARY)
    coordinates = []
    for y in range(thresh.shape[0]):
        for x in range(thresh.shape[1]):
            if thresh[y,x] > 0:
                coordinates.append([x*8, y*8])
                # coordinates.append([x, y])

    neighbors = {} # Get neighbors by index
    d_sq = 128 # distance thresh
    for i, xy in enumerate(coordinates): 
        x, y = xy[0], xy[1]
        neighbors[i] = []
        for j in range(i+1, len(coordinates)):
            next_xy = coordinates[j]
            next_x, next_y = next_xy[0], next_xy[1]
            if (x-next_x)**2 + (y-next_y)**2 <= d_sq:
                neighbors[i].append(j)
    return coordinates, neighbors

def main():

    global with_warp, blur_window_size
    xys, neighbors = get_tracking_points()

    method = cv.TM_CCORR_NORMED
    half_template_size = 18 #50#18#150 #18
    # x, y = (439, 353)
    # x, y = x*640/1200, y*640/1200
    # xys = [[584, 240], [565, 266], [534, 285], [487, 321], [439, 353], [402, 389], [412, 448], \
    # 		[458, 484], [490, 510], [548, 556], [569, 576], [600, 622], [635, 585], [668, 490], \
    # 		[686, 424], [711, 369], [751, 342], [756, 318], [716, 271], [634, 288]]
    num_pts = len(xys)
    for i, xy in enumerate(xys): 
        x, y = xy[0], xy[1]
        # x, y = x*640/1200, y*640/1200
        xys[i] = [x, y]
    half_source_size = 25 #60#25#160#25#25

    xy_list = np.zeros((len(xys), 28, 2))
    xy_list[:,0,:] = xys

    lost_pt = []
    for ii, i in enumerate(range(4051,4072)): #4078
        f1 = str(i)+'.jpg'
        f2 = str(i+1)+'.jpg'
        print('----')
        print(f2)

        ori_file = ori_path + f2
        ori_prev = ori_path + f1

        prev_img = cv.imread(ori_prev)
        prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
        next_img = cv.imread(ori_file)
        next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
        if with_warp: 
            prev_img = cv.remap(prev_img, warp_x, warp_y, cv.INTER_LINEAR)
            next_img = cv.remap(next_img, warp_x, warp_y, cv.INTER_LINEAR)
            prev_gray = cv.remap(prev_gray, warp_x, warp_y, cv.INTER_LINEAR)
            next_gray = cv.remap(next_gray, warp_x, warp_y, cv.INTER_LINEAR)

        plt.figure(figsize=(21, 12))
        plt.subplot(131)
        plt.imshow(next_img)
        plt.subplot(132)
        
        next_gaussian = cv.GaussianBlur(next_gray,(blur_window_size,blur_window_size),0)

        plt.imshow(next_gaussian, cmap='gray')

        for j in range(num_pts):

            if j not in lost_pt: 

                x, y = int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])

                next_match_x, next_match_y = find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)

                if next_match_x == np.Inf: 
                    lost_pt.append(j)

                else:
                    x, y = next_match_x, next_match_y

                    xy_list[j,ii+1,:] =[x, y] 

                    first_x, first_y = xy_list[j, 0, 0], xy_list[j, 0, 1] 
                    # cv.circle(next_img, (int(first_x), int(first_y)), 3, (0, 255, 0))

                    prev_x, prev_y = xy_list[j, ii, 0], xy_list[j, ii, 1] 
                    cv.arrowedLine(next_img, (int(prev_x), int(prev_y)), \
                                (int(x), int(y)), (255,0,0), 1, tipLength=0.3)
                    # cv.circle(next_img, (int(prev_x), int(prev_y)), 3, (255, 0, 0))
                    cv.circle(next_img, (int(next_match_x), int(next_match_y)), 3, (0, 255, 0))

        # plt.imsave('../images/gaussian_blur/'+f2, next_img)
        plt.subplot(133)
        plt.imshow(next_img)
        # pdb.set_trace()
        plt.savefig('../images/gaussian_blur/'+f2)
    print("compare first and last of cycle")
    # pdb.set_trace()
    f1 = '4052.jpg'
    f2 = '4071.jpg'

    ori_file = ori_path + f2
    ori_prev = ori_path + f1

    print(ori_prev)
    print(ori_file)

    prev_img = cv.imread(ori_prev)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    next_img = cv.imread(ori_file)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
    if with_warp:
        prev_img = cv.remap(prev_img, warp_x, warp_y, cv.INTER_LINEAR)
        next_img = cv.remap(next_img, warp_x, warp_y, cv.INTER_LINEAR)
        prev_gray = cv.remap(prev_gray, warp_x, warp_y, cv.INTER_LINEAR)
        next_gray = cv.remap(next_gray, warp_x, warp_y, cv.INTER_LINEAR)
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(next_img)

    for j in range(num_pts):

        x, y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])
        next_match_x, next_match_y = find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)

        if next_match_x != np.Inf: 
            x, y = next_match_x, next_match_y

            xy_list[j,0+1,:] =[x, y] 

            first_x, first_y = xy_list[j, 0, 0], xy_list[j, 0, 1] 
            # cv.circle(next_img, (int(first_x), int(first_y)), 3, (0, 255, 0))

            prev_x, prev_y = xy_list[j, 0, 0], xy_list[j, 0, 1] 
            cv.arrowedLine(next_img, (int(prev_x), int(prev_y)), \
                        (int(x), int(y)), (255,0,0), 1, tipLength=0.3)
            # cv.circle(next_img, (int(prev_x), int(prev_y)), 3, (255, 0, 0))
            cv.circle(next_img, (int(next_match_x), int(next_match_y)), 3, (255, 255, 0))
    # plt.imsave('../images/gaussian_blur/4052_4071.jpg', next_img)
    plt.subplot(122)
    plt.imshow(next_img)
    plt.savefig('../images/gaussian_blur/4052_4071.jpg')

if __name__ == "__main__":
    # execute only if run as a script
    main()
