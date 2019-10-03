import cv2 as cv
from block_matching import block_match
import pdb
import glob
import matplotlib.pyplot as plt
import numpy as np

## Global variables 
path16 = '../images/downsamplePlot/down_by_16/'
files16 = glob.glob(path16+"*.jpg")
files16 = sorted(files16)
path8 = '../images/downsamplePlot/down_by_8/'
files8 = glob.glob(path8+"*.jpg")
files8 = sorted(files8)
path4 = '../images/downsamplePlot/down_by_4/'
files4 = glob.glob(path4+"*.jpg")
files4 = sorted(files4)
path2 = '../images/downsamplePlot/down_by_2/'
files2 = glob.glob(path2+"*.jpg")
files2 = sorted(files2)
ori_path = '../images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)


def find_match(prev_file, next_file, ds_factor, x, y, prev_x, prev_y, half_template_size, method, first_downsize_by):
    global path16, path8, path4, path2, ori_path

    if prev_x == np.Inf: 
        half_source_size = prev_y
        prev_x = x
        prev_y = y
    else:
        half_source_size = half_template_size * 2

    x, y = x*first_downsize_by/ds_factor, y*first_downsize_by/ds_factor


    ori_file = ori_path + next_file
    ori_prev = ori_path + prev_file

    if ds_factor == 16: 
        prev_file_path = path16+prev_file
        next_file_path = path16+next_file 
    elif ds_factor == 8: 
        prev_file_path = path8+prev_file
        next_file_path = path8+next_file 
    elif ds_factor == 4: 
        prev_file_path = path4+prev_file
        next_file_path = path4+next_file 
    elif ds_factor == 2: 
        prev_file_path = path2+prev_file
        next_file_path = path2+next_file 
    elif ds_factor == 1: 
        prev_file_path = ori_prev
        next_file_path = ori_file


    prev_img = cv.imread(prev_file_path)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
 
    next_img = cv.imread(next_file_path)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)

    template = prev_gray[y-half_template_size:y+half_template_size+1, x-half_template_size:x+half_template_size+1]
    w, h = template.shape[::-1] 
    w, h = w-1, h-1
    ## define source around the previous match
    yy_ul, yy_lr = prev_y-half_source_size, prev_y+half_source_size+1
    xx_ul, xx_lr =  prev_x-half_source_size, prev_x+half_source_size+1

    if yy_ul < 0: 
        yy_ul = 0
    if xx_ul < 0: 
        xx_ul = 0
    if yy_lr > prev_gray.shape[0]-1:
        yy_lr = prev_gray.shape[0]-1
    if xx_lr > prev_gray.shape[1]-1: 
        xx_lr = prev_gray.shape[1]-1

    source = next_gray[yy_ul:yy_lr, xx_ul:xx_lr]
    
    (next_match_x, next_match_y) = block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y)
    return (next_match_x, next_match_y)

def find_match_hierarchy(prev_file, next_file, x, y, half_template_size, method, match_dictionary, ori_img, first_downsize_by=4, plotit=True):

    if (first_downsize_by != 4):
        print("Need manual modification to code")
        raise ValueError
    
    found_match = False
    # ## downsampled by 8
        # (next_match_x, next_match_y) = find_match(prev_file, next_file, 8, x, y, np.Inf, 6 , half_template_size, method)
        # if next_match_x != np.Inf:
        #     ## downsampled by 4
        #     next_match_x, next_match_y = next_match_x * 2, next_match_y * 2
        #     (next_match_x, next_match_y) = find_match(prev_file, next_file, 4, x, y, next_match_x, next_match_y, half_template_size, method)
    (next_match_x, next_match_y) = find_match(prev_file, next_file, 4, x, y, np.Inf, 6, half_template_size, method, first_downsize_by)

    if next_match_x != np.Inf:
        ## downsampled by 2
        next_match_x, next_match_y = next_match_x * 2, next_match_y * 2
        (next_match_x, next_match_y) = find_match(prev_file, next_file, 2, x, y, next_match_x, next_match_y, half_template_size, method, first_downsize_by)

        if next_match_x != np.Inf:
            # ## in original image
            next_match_x, next_match_y = next_match_x * 2, next_match_y * 2

            (next_match_x, next_match_y) = find_match(prev_file, next_file, 1, x, y, next_match_x, next_match_y, half_template_size, method, first_downsize_by)

            if next_match_x != np.Inf:
                # print("found a match")
                ## Draw displacement field on the original image 
                next_match_y, next_match_x = int(next_match_y), int(next_match_x)
                
                displacement = np.sqrt((x*first_downsize_by-next_match_x)**2+(y*first_downsize_by-next_match_y)**2)
                if displacement <= 20: #5+15.*y/next_img.shape[0]:
                    found_match = True
                    
                    match_dictionary[(next_match_x, next_match_y)] = (x*first_downsize_by, y*first_downsize_by)
                    if plotit:
                        # color = helper_get_color(next_match_x, next_match_y, x*first_downsize_by, y*first_downsize_by)
                        # cv.arrowedLine(ori_img, (next_match_x, next_match_y), (x*first_downsize_by,y*first_downsize_by), color, 1, tipLength=0.3)
                        
                        color= (0,0,255)

                        cv.line(ori_img, (x*first_downsize_by,y*first_downsize_by), (next_match_x, next_match_y), color, 1)
                        # cv.circle(ori_img, (next_match_x, next_match_y), 3, (0, 255, 0))
                        # print('green: '+str((next_match_x, next_match_y)))


    if (found_match == False) & (plotit== True):
        cv.circle(ori_img, (x*first_downsize_by, y*first_downsize_by), 1, (255, 0, 0))
        print('no match')
    return ori_img, match_dictionary, found_match, next_match_x, next_match_y


def __main__():
    
    f1 = '4067.jpg'
    f2 = '4068.jpg'

    match_dictionary = {}
    method = cv.TM_CCORR_NORMED
    half_template_size = 4
    x, y = 100, 45 #50,70. 55,70
    first_downsize_by = 4

    xy_list = [[x,y]]

    for i in range(4051,4070): #4078
        ori_img = cv.imread(ori_path+f2)
        print('----')
        print(f2)
        f1 = str(i)+'.jpg'
        f2 = str(i+1)+'.jpg'
        ori_img, match_dictionary, found_match, next_match_x, next_match_y = find_match_hierarchy(f1, f2, x, y, half_template_size, method, match_dictionary, ori_img)
        if found_match != True: 
            print(found_match)
        else:
            # match = match_dictionary.keys()[0]
            # x, y = match[0]/4, match[1]/4
            x, y = next_match_x/first_downsize_by, next_match_y/first_downsize_by
        print(x,y)
        xy_list.append([x,y])

        for j, xy in enumerate(xy_list): 
            if j <= len(xy_list):
                prev_x, prev_y = xy_list[j-1][0], xy_list[j-1][1] 
                new_x, new_y = xy[0], xy[1]
                cv.line(ori_img, (int(prev_x*first_downsize_by), int(prev_y*first_downsize_by)), \
                        (int(new_x*first_downsize_by), int(new_y*first_downsize_by)), (0,0,255), 1)
                cv.circle(ori_img, (int(new_x*first_downsize_by), int(new_y*first_downsize_by)), 3, (255, 255, 0))
        cv.circle(ori_img, (int(new_x*first_downsize_by), int(new_y*first_downsize_by)), 3, (255, 0, 0))
        # print('red: '+str((x*first_downsize_by, y*first_downsize_by)))

        fig = plt.figure(figsize=(14,8),frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(ori_img)
        # plt.show()
        plt.savefig('../images/3hierarchy_ccorr_normed/1pt_trace/'+f2)

    pdb.set_trace()