import cv2 as cv
from block_matching import block_match
from trace_trajectory import find_match
import pdb
import glob
import matplotlib.pyplot as plt
import numpy as np

## Global variables 
path16 = '../images/downsamplePlot/down_by_16/'
path8 = '../images/downsamplePlot/down_by_8/'
path4 = '../images/downsamplePlot/down_by_4/'
path2 = '../images/downsamplePlot/down_by_2/'
ori_path = '../images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

def find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method):



	blur_window_size = 9
	prev_gaussian = cv.GaussianBlur(prev_gray,(blur_window_size,blur_window_size),0)
	next_gaussian = cv.GaussianBlur(next_gray,(blur_window_size,blur_window_size),0)

	## define template
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
	(next_match_x, next_match_y) = block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y)

	return next_match_x, next_match_y

def get_tracking_points():
    '''Get points of interest based on intensity in downsampled by 8 images'''
    f = path8 + '4051.jpg'
    img = cv.imread(f)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray,80,1,cv.THRESH_BINARY)
    coordinates = []
    for y in range(thresh.shape[0]):
        for x in range(thresh.shape[1]):
            if thresh[y,x] > 0:
                coordinates.append([x*8, y*8])
    return coordinates


def main():

    xys = get_tracking_points()

    method = cv.TM_CCORR_NORMED
    half_template_size = 18
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
    half_source_size = 25

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
        # pdb.set_trace()
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
                    cv.circle(next_img, (int(next_match_x), int(next_match_y)), 3, (255, 255, 0))

        plt.imsave('../images/gaussian_blur/'+f2, next_img)

    print("compare first and last of cycle")
    pdb.set_trace()
    f1 = '4052.jpg'
    f2 = '4071.jpg'

    ori_file = ori_path + f2
    ori_prev = ori_path + f1

    prev_img = cv.imread(ori_prev)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    next_img = cv.imread(ori_file)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
    for j in range(num_pts):
        x, y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])

        next_match_x, next_match_y = find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)
        x, y = next_match_x, next_match_y

        xy_list[j,0+1,:] =[x, y] 

        first_x, first_y = xy_list[j, 0, 0], xy_list[j, 0, 1] 
        prev_x, prev_y = xy_list[j, 0, 0], xy_list[j, 0, 1] 
        cv.arrowedLine(next_img, (int(first_x), int(first_y)), \
                    (int(x), int(y)), (0,0,255), 1, tipLength=0.3)
        cv.circle(next_img, (int(prev_x), int(prev_y)), 3, (255, 255, 0))
        cv.circle(next_img, (int(next_match_x), int(next_match_y)), 3, (255, 0, 0))
    plt.imsave('../images/gaussian_blur/4052_4071.jpg', next_img)


if __name__ == "__main__":
    # execute only if run as a script
    main()
