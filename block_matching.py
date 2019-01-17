''' Block matching to the neighboring pixels'''

import cv2 as cv
import numpy as np
import pdb
import glob
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

from hog import hog_match

## Global variables 
path16 = 'images/downsamplePlot/down_by_16/'
files16 = glob.glob(path16+"*.jpg")
files16 = sorted(files16)
path8 = 'images/downsamplePlot/down_by_8/'
files8 = glob.glob(path8+"*.jpg")
files8 = sorted(files8)
path4 = 'images/downsamplePlot/down_by_4/'
files4 = glob.glob(path4+"*.jpg")
files4 = sorted(files4)
path2 = 'images/downsamplePlot/down_by_2/'
files2 = glob.glob(path2+"*.jpg")
files2 = sorted(files2)
ori_path = 'images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)



def block_matching_cv(template_gray, source_gray,  method=cv.TM_CCORR): 
    if len(template_gray.shape) > 2 or len(source_gray.shape) > 2: 
        template_gray = cv.cvtColor(template_gray, cv.COLOR_BGR2GRAY) 
        source_gray = cv.cvtColor(source_gray, cv.COLOR_BGR2GRAY) 

    # Perform match operations
    try:
        res = cv.matchTemplate(source_gray, template_gray, method) 
    except cv.error: 
        return []
    # thresh=0.8
    thresh=0.1
    maxval = max(res.max(), thresh)
    loc = np.where( res >= maxval) 
    # pdb.set_trace()

    # try: 
    #     match_loc =zip(*loc[::-1])
    #     pt = match_loc[0]
    # except: 
    #     pdb.set_trace()
    
    # # Draw a rectangle around the matched region. 
    # w, h = template_gray.shape[::-1] 
    # for pt in zip(*loc[::-1]): 
    #     print(pt)
    #     cv.rectangle(source_gray, pt, (match_ul_y + w, match_ul_x + h), 255, 1) 

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(template_gray)
    # plt.subplot(122)
    # plt.imshow(source_gray)
    # plt.show()

    return loc


def block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y): 
    prev_match_x, prev_match_y = np.Inf, np.Inf
    if method == 'hog': 
        loc = hog_match(template, source)
    else: 
        loc = block_matching_cv(template, source)

    if method == 'hog': 
        match_loc = loc
    else: 
        match_loc = zip(*loc[::-1])

    if len(match_loc)>0:
        
        try:
            pt = match_loc[0]
        except:
            pdb.set_trace()
        match_ul_y, match_ul_x = pt[0], pt[1]

        prev_match_y = yy_ul + match_ul_y + half_template_size
        prev_match_x = xx_ul + match_ul_x + half_template_size
            
    return (prev_match_x, prev_match_y)


def find_match(prev_file, next_file, ds_factor, x, y, prev_x, prev_y, half_template_size, method):
    global path16, path8, path4, path2, ori_path

    half_source_size = half_template_size * 2
    x, y = x*4/ds_factor, y*4/ds_factor

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

def plot_match(prev_file, next_file, half_template_size=7, method = 'hog'): 

    '''
    Inputs: 
    - prev_file: (possibly downsampled)image file to previous frame
    - next_file: (possibly downsampled) image file to current frame of interest
    - ori_file: the original image file to the current frameof interest
    - ds_factor: downsampling factor
    '''
    global path16, path8, path4, path2, ori_path
    half_source_size = half_template_size * 2

    ori_file = ori_path + next_file
    ori_prev = ori_path + prev_file

    prev_img = cv.imread(path4+prev_file)
    next_img = cv.imread(path4+next_file)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)

    ori_img = cv.imread(ori_file)
    ori_prev = cv.imread(ori_prev)

    # coordinates = peak_local_max(next_gray, threshold_abs = 100, min_distance=3) # footprint=np.ones((5,5)) for original
    _, thresh = cv.threshold(next_gray,100,1,cv.THRESH_BINARY)
    coordinates = []
    for x in range(thresh.shape[0]):
        for y in range(thresh.shape[1]):
            if thresh[x,y] > 0:
                coordinates.append([x,y])
    
    ## define template by around a peak 
    print('total', len(coordinates))
    for counter, xy in enumerate(coordinates):
        
        if counter %50==0: 
            print(counter)
        
        # xy = coordinates[0]
        try:
            x, y = xy[0], xy[1]
        except:
            print(xy)
            pdb.set_trace()
        ## downsampled by 4
        (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 4, x, y, x,y , half_template_size, method)
        if prev_match_x != np.Inf:
            # print("match step 1")
            ## downsampled by 2
            prev_match_x, prev_match_y = prev_match_x * 2, prev_match_y * 2
            (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 2, x, y, prev_match_x, prev_match_y, half_template_size, method)

            if prev_match_x != np.Inf:
                # print("matched step 2")
                ## in original image
                prev_match_x, prev_match_y = prev_match_x * 2, prev_match_y * 2

                prev_match_x_step2, prev_match_y_step2 = prev_match_x, prev_match_y

                (prev_match_x, prev_match_y) = find_match(prev_file, next_file, 1, x, y, prev_match_x, prev_match_y, half_template_size, method)

                if prev_match_x != np.Inf:
                    # print("found a match")
                    ## Draw displacement field on the original image 
                    prev_match_y, prev_match_x = int(prev_match_y), int(prev_match_x)
                    
                    cv.arrowedLine(ori_img, (prev_match_y, prev_match_x), (y*4,x*4), (255,255,0), 1, tipLength=0.3)
                    cv.circle(ori_img, (y*4, x*4), 2, (255, 0, 0))


    ## Save displacement field in figure
    if method == 'hog': 
        savefilename = 'images/demo_3hierarchy_bm/'+'HOG/'+next_file[-8::]
    else:
        savefilename = 'images/demo_3hierarchy_bm/'+'TM_CCORR/'+next_file[-8::]

    plt.figure(figsize=(15,12))
    plt.imshow(ori_img)
    plt.savefig(savefilename)


# path = 'images/downsamplePlot/down_by_16/'
# files = glob.glob(path+"*.jpg")
# files = sorted(files)
# ori_path = 'images/original/'
# ori_files = glob.glob(ori_path+"*.jpg")
# ori_files = sorted(ori_files)

# # file1 = files[0]
# # file2 = files[1]
# # plot_match(file1, file2)

# for i, file in enumerate(files): 
#     print(file)
#     if i < len(files)-1: 
#         plot_match(file, files[i+1], ori_files[i+1], 16)






# file1 = path+'4063.jpg'
# file2 = path+'4064.jpg'
# ori1 = ori_path+'4063.jpg'
# ori2 = ori_path+'4064.jpg'

f1 = '4067.jpg'
f2 = '4068.jpg'
# plot_match(f1, f2, method=cv.TM_CCORR)
plot_match(f1, f2, method='hog')

# for i, file in enumerate(files): 
#     print(file[-8::])
#     if i < len(files)-1: 
#         plot_match(file, files[i+1], ori_files[i], ori_files[i+1],8, half_template_size=15, half_source_size=17)


# path = 'images/downsamplePlot/down_by_4/'
# files = glob.glob(path+"*.jpg")
# files = sorted(files)
# ori_path = 'images/original/'
# ori_files = glob.glob(ori_path+"*.jpg")
# ori_files = sorted(ori_files)

# for i, file in enumerate(files): 
#     print(file[-8::])
#     if i < len(files)-1: 
#         plot_match(file, files[i+1], ori_files[i], 4, half_template_size=3, half_source_size=5 )

