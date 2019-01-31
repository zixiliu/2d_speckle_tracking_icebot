''' Block matching to the neighboring pixels'''

import cv2 as cv
import numpy as np
import pdb
import glob
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

from hog import hog_match
import sys
sys.path.append("..")

from python_local_maxima_block_matching.remove_outlier import remove_outlier
from python_local_maxima_block_matching.pt_matching import priv_draw_displacement, helper_get_color

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



def block_matching_cv(template_gray, source_gray,  method): 
    if len(template_gray.shape) > 2 or len(source_gray.shape) > 2: 
        template_gray = cv.cvtColor(template_gray, cv.COLOR_BGR2GRAY) 
        source_gray = cv.cvtColor(source_gray, cv.COLOR_BGR2GRAY) 

    # Perform match operations
    try:
        res = cv.matchTemplate(source_gray, template_gray, method) 
    except cv.error: 
        return []
    thresh = 0.9
    maxval = max(res.max(), thresh)
    # maxval = res.max()
    loc = np.where( res >= maxval) 
    try:
        y,x = loc[0][0], loc[1][0]
        loc = [[y, x]]
    except:
        print(res.max())
        return []
    return loc


def block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y, \
                template_yy_ul, template_yy_lr, template_xx_ul, template_xx_lr): 
    prev_match_x, prev_match_y = np.Inf, np.Inf
    if method == 'hog': 
        loc = hog_match(template, source)
    else: 
        loc = block_matching_cv(template, source, method)

    if len(loc)>0:
        
        try:
            pt = loc[0]
        except:
            pdb.set_trace()
        match_ul_y, match_ul_x = pt[0], pt[1]

        prev_match_y = yy_ul + match_ul_y + (y-template_yy_ul)
        prev_match_x = xx_ul + match_ul_x + (x-template_xx_ul)
            
    return (prev_match_x, prev_match_y)


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

    template = next_gray[y-half_template_size:y+half_template_size+1, x-half_template_size:x+half_template_size+1]
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

    source = prev_gray[yy_ul:yy_lr, xx_ul:xx_lr]
    
    (prev_match_x, prev_match_y) = block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y)
    return (prev_match_x, prev_match_y)

def find_match_hierarchy(prev_file, next_file, x, y, half_template_size, method, match_dictionary, ori_img, first_downsize_by=4, plotit=True):

    if (first_downsize_by != 4):
        print("Need manual modification to code")
        raise ValueError
    
    found_match = False
    # ## downsampled by 8
        # (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 8, x, y, np.Inf, 6 , half_template_size, method)
        # if prev_match_x != np.Inf:
        #     ## downsampled by 4
        #     prev_match_x, prev_match_y = prev_match_x * 2, prev_match_y * 2
        #     (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 4, x, y, prev_match_x, prev_match_y, half_template_size, method)
    (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 4, x, y, np.Inf, 6, half_template_size, method, first_downsize_by)

    if prev_match_x != np.Inf:
        ## downsampled by 2
        prev_match_x, prev_match_y = prev_match_x * 2, prev_match_y * 2
        (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 2, x, y, prev_match_x, prev_match_y, half_template_size, method, first_downsize_by)

        if prev_match_x != np.Inf:
            # ## in original image
            prev_match_x, prev_match_y = prev_match_x * 2, prev_match_y * 2

            (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 1, x, y, prev_match_x, prev_match_y, half_template_size, method, first_downsize_by)

            if prev_match_x != np.Inf:
                # print("found a match")
                ## Draw displacement field on the original image 
                prev_match_y, prev_match_x = int(prev_match_y), int(prev_match_x)
                
                displacement = np.sqrt((x*first_downsize_by-prev_match_x)**2+(y*first_downsize_by-prev_match_y)**2)
                if displacement <= 20: #5+15.*y/next_img.shape[0]:
                    found_match = True
                    
                    match_dictionary[(prev_match_x, prev_match_y)] = (x*first_downsize_by, y*first_downsize_by)
                    if plotit:
                        color = helper_get_color(prev_match_x, prev_match_y, x*first_downsize_by, y*first_downsize_by)
                        # cv.arrowedLine(ori_img, (prev_match_x, prev_match_y), (x*first_downsize_by,y*first_downsize_by), color, 1, tipLength=0.3)
                        
                        color= (0,0,255)

                        cv.line(ori_img, (prev_match_x, prev_match_y), (x*first_downsize_by,y*first_downsize_by), color, 1)
                        cv.circle(ori_img, (x*first_downsize_by, y*first_downsize_by), 3, (0, 255, 0))
                        print('plot')
    if (found_match == False) & (plotit== True):
        cv.circle(ori_img, (x*first_downsize_by, y*first_downsize_by), 1, (255, 0, 0))
        print('no match')
    return ori_img, match_dictionary, found_match

def plot_match(prev_file, next_file, half_template_size=7, method = 'hog'): 

    '''
    Inputs: 
    - prev_file: (possibly downsampled)image file to previous frame
    - next_file: (possibly downsampled) image file to current frame of interest
    - ori_file: the original image file to the current frameof interest
    - ds_factor: downsampling factor
    '''
    global path16, path8, path4, path2, ori_path
    
    first_downsize_by = 4


    match_dictionary = {}


    half_source_size = half_template_size * 2

    ori_file = ori_path + next_file
    ori_prev = ori_path + prev_file
    ori_img = cv.imread(ori_file)
    ori_prev = cv.imread(ori_prev)

    if first_downsize_by == 16: 
        prev_file_path = path16+prev_file
        next_file_path = path16+next_file 
    elif first_downsize_by == 8: 
        prev_file_path = path8+prev_file
        next_file_path = path8+next_file 
    elif first_downsize_by == 4: 
        prev_file_path = path4+prev_file
        next_file_path = path4+next_file 
    elif first_downsize_by == 2: 
        prev_file_path = path2+prev_file
        next_file_path = path2+next_file 
    elif first_downsize_by == 1: 
        prev_file_path = ori_prev
        next_file_path = ori_file


    prev_img = cv.imread(prev_file_path)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
 
    next_img = cv.imread(next_file_path)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)

    

    # coordinates = peak_local_max(next_gray, threshold_abs = 100, min_distance=3) # footprint=np.ones((5,5)) for original
    _, thresh = cv.threshold(next_gray,70,1,cv.THRESH_BINARY)
    coordinates = []
    for y in range(thresh.shape[0]):
        for x in range(thresh.shape[1]):
            if thresh[y,x] > 0:
                coordinates.append([y, x])
    # for x in np.arange(10*8/first_downsize_by,55*8/first_downsize_by+1, 8/first_downsize_by): 
    #     for y in np.arange(10*8/first_downsize_by,70*8/first_downsize_by+1, 8/first_downsize_by): 
    #         coordinates.append([x,y])


    
    ## define template by around a peak 
    print('total', len(coordinates))
    for counter, xy in enumerate(coordinates):
        
        # if counter %50==0: 
        #     print(counter)
        
        # xy = coordinates[0]
        try:
            x, y = xy[1], xy[0]
        except:
            print(xy)
            pdb.set_trace()        

        ori_img, match_dictionary, _ = find_match_hierarchy(prev_file, next_file, x, y, half_template_size, method, match_dictionary, ori_img)


    ## Save displacement field in figure
    # if method == 'hog': 
    #     savefilename = '../images/demo_3hierarchy_bm/'+'HOG/'+'pre_outlier_rm_'+next_file[-8::]
    # elif method == cv.TM_CCORR:
    #     savefilename = '../images/demo_3hierarchy_bm/'+'TM_CCORR/'+'pre_outlier_rm_'+next_file[-8::]
    # elif method == cv.TM_CCORR_NORMED:
    #     savefilename = '../images/demo_3hierarchy_bm/'+'TM_CCORR_NORMED/'+'pre_outlier_rm_'+next_file[-8::]


    savefilename = '../images/3hierarchy_ccorr_normed/'+'pre_outlier_rm_'+next_file[-8::]

    fig = plt.figure(figsize=(20,20),frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(ori_img)
    plt.savefig(savefilename) 
    # plt.imsave(savefilename, ori_img)
    print("Block Matching Done!")
    return match_dictionary



# path = '../images/downsamplePlot/down_by_16/'
# files = glob.glob(path+"*.jpg")
# files = sorted(files)
# ori_path = '../images/original/'
# ori_files = glob.glob(ori_path+"*.jpg")
# ori_files = sorted(ori_files)

# # file1 = files[0]
# # file2 = files[1]
# # plot_match(file1, file2)

# for i, file in enumerate(files): 
#     print(file)
#     if i < len(files)-1: 
#         plot_match(file, files[i+1], ori_files[i+1], 16)




# f1 = '4067.jpg'
# f2 = '4068.jpg'
# match_dictionary = plot_match(f1, f2, method=cv.TM_CCORR_NORMED, half_template_size = 4)

# for i in range(4051,4078):
#     f1 = str(i)+'.jpg'
#     f2 = str(i+1)+'.jpg'
#     print('------------------------------------------------')
#     print(f2)
#     match_dictionary = plot_match(f1, f2, method=cv.TM_CCORR_NORMED, half_template_size = 4)

#     new_img = cv.imread(ori_path+f2)
#     match_dict, new_img = remove_outlier(match_dictionary, new_img.shape, new_img, method='neighbor_pixel')

#     # # Convert match dictionary to a list
#     match_list = [[x,match_dict[x]] for x in match_dict.keys()]
#     priv_draw_displacement(match_list, ori_path+f1, ori_path+f2, new_img)

# pdb.set_trace()

# for i, file in enumerate(files): 
#     print(file[-8::])
#     if i < len(files)-1: 
#         plot_match(file, files[i+1], ori_files[i], ori_files[i+1],8, half_template_size=15, half_source_size=17)


# path = '../images/downsamplePlot/down_by_4/'
# files = glob.glob(path+"*.jpg")
# files = sorted(files)
# ori_path = '../images/original/'
# ori_files = glob.glob(ori_path+"*.jpg")
# ori_files = sorted(ori_files)

# for i, file in enumerate(files): 
#     print(file[-8::])
#     if i < len(files)-1: 
#         plot_match(file, files[i+1], ori_files[i], 4, half_template_size=3, half_source_size=5 )

