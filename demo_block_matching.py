''' Block matching to the neighboring pixels'''

import cv2 as cv
import numpy as np
import pdb
import glob
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

from hog import hog_match


def block_matching(template_gray, source_gray,  method, savefilename): 
    if len(template_gray.shape) > 2 or len(source_gray.shape) > 2: 
        template_gray = cv.cvtColor(template_gray, cv.COLOR_BGR2GRAY) 
        source_gray = cv.cvtColor(source_gray, cv.COLOR_BGR2GRAY) 

    # Perform match operations
    try:
        res = cv.matchTemplate(source_gray, template_gray, method) 
    except cv.error: 
        return []
    thresh=0.8
    maxval = max(res.max(), thresh)
    loc = np.where( res >= maxval) 

    y,x = loc[0][0], loc[1][0]
    loc = [[y, x]]
    # pdb.set_trace()
    
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

    ## Draw all the blocks 
    print(template_gray.shape, source_gray.shape)
    template_width = template_gray.shape[0]
    source_width = source_gray.shape[0]

    temp = source_width - template_width + 1

    print(loc)
    print(res)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(template_gray, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(source_gray, cmap='gray')
    # plt.savefig(savefilename+'_1.png')


    fig = plt.figure()

    for i in range(temp * temp):
        # plt.subplot(440+i+1)
        fig.add_subplot(temp,temp,i+1)
        subimg = source_gray[i/temp:i/temp+template_width, i%temp:i%temp+template_width]
        plt.imshow(subimg, cmap='gray')
    plt.savefig(savefilename+'_2.png')

    # plt.show()

    # pdb.set_trace()
    

    return loc

def find_match(source, template, next_img, prev_img, next_gray, prev_gray, ori_img, ori_prev, ds_factor, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y, savefilename): 
    half_source_size = half_template_size *2 

    w, h = template.shape[::-1] 
    w, h = w-1, h-1

    if method == 'hog': 
        loc = hog_match(template, source)
    else: 
        loc = block_matching(template, source, method, savefilename)


    next_img[x,y] = np.array([0,0,255]) 

    cv.circle(ori_img, (y*ds_factor, x*ds_factor), 1, (255,0,0),1)

    if len(loc)>0:
        # cv.circle(next_img, (y, x), 1, (255, 0, 0))
        # cv.rectangle(ori, (y-half_source_size, x-half_source_size),(y+half_source_size, x+half_source_size), (0,255,0), 1) 
        # cv.rectangle(next_img, (y-half_template_size, x-half_template_size),(y+half_template_size, x+half_template_size), (0,255,0), 1)    
        # print('new', (y-half_template_size, x-half_template_size),(y+half_template_size, x+half_template_size))

        # # Draw a rectangle around the matched region. 

        
        # print("match_loc", match_loc)

        for pt in loc: 
            
            match_ul_y, match_ul_x = pt[0], pt[1]

            ## Matched block center in the previous frame
            # prev_match_y = y - half_source_size + half_template_size + match_ul_y
            # prev_match_x = x - half_source_size + half_template_size + match_ul_x
            prev_match_y = yy_ul + match_ul_y + half_template_size
            prev_match_x = xx_ul + match_ul_x + half_template_size
            prev_img[prev_match_x, prev_match_y] = np.array([0,0,255])
            print("\n")
            print('prev_match_x,y')
            print(prev_match_x, prev_match_y)
    
            print('source xx_ul, xx_lr, yy_ul, yy_lr,')
            print(xx_ul, xx_lr, yy_ul, yy_lr)
            

            
            # cv.rectangle(prev_img, (y-half_source_size+match_ul_y, x-half_source_size+match_ul_x), (y-half_source_size+match_ul_y + w, x-half_source_size+match_ul_x + h), (0,0,255), 1)
            # prev_img[prev_match_y, prev_match_x] = np.array([0,0,255])
            # print('prev', (y-half_source_size+match_ul_y, x-half_source_size+match_ul_x), (y-half_source_size+match_ul_y + w, x-half_source_size+match_ul_x + h)) 
            # cv.arrowedLine(next_img, (prev_match_y+1, prev_match_x+1), (y,x), (255,255,0), 1, tipLength=0.3)
           
            ## From previous frame to current frame
            cv.arrowedLine(ori_img, (prev_match_x*ds_factor, prev_match_y*ds_factor), \
                    (x*ds_factor,y*ds_factor), (255,255,0), 1, tipLength=0.3) 
            cv.arrowedLine(ori_prev, (prev_match_x*ds_factor, prev_match_y*ds_factor), \
                    (x*ds_factor,y*ds_factor), (255,255,0), 1, tipLength=0.3) 

            cv.circle(ori_img, (x*ds_factor, y*ds_factor), 2, (255, 0, 0))
            cv.circle(ori_img, (prev_match_x*ds_factor, prev_match_y*ds_factor), 2, (0, 0, 255))
            cv.circle(ori_prev, (x*ds_factor, y*ds_factor), 2, (255, 0, 0))
            cv.circle(ori_prev, (prev_match_x*ds_factor, prev_match_y*ds_factor), 2, (0, 0, 255))
            # cv.arrowedLine(ori_img, (prev_match_y*ds_factor, prev_match_x*ds_factor), \
            #         (y*ds_factor,x*ds_factor), (255,255,0), 1, tipLength=0.3) 
            # cv.arrowedLine(ori_prev, (prev_match_y*ds_factor, prev_match_x*ds_factor), \
            #         (y*ds_factor,x*ds_factor), (255,255,0), 1, tipLength=0.3) 

            # cv.circle(ori_img, (y*ds_factor, x*ds_factor), 2, (255, 0, 0))
            # cv.circle(ori_img, (prev_match_y*ds_factor, prev_match_x*ds_factor), 2, (0, 0, 255))
            # cv.circle(ori_prev, (y*ds_factor, x*ds_factor), 2, (255, 0, 0))
            # cv.circle(ori_prev, (prev_match_y*ds_factor, prev_match_x*ds_factor), 2, (0, 0, 255))

            match_upper_left_y = (prev_match_y-half_template_size)*ds_factor
            match_upper_left_x = (prev_match_x-half_template_size)*ds_factor
            
            match_lower_right_y = (prev_match_y+half_template_size)*ds_factor
            match_lower_right_x = (prev_match_x+half_template_size)*ds_factor
            
            # cv.rectangle(ori_img, (match_upper_left_y,match_upper_left_x), (match_lower_right_y,match_lower_right_x), (0,0,255), 1)
            # cv.rectangle(ori_img, ((y-half_template_size)*ds_factor,(x-half_template_size)*ds_factor), ((y+half_template_size)*ds_factor,(x+half_template_size)*ds_factor), (255,0,0), 1) 
            # cv.rectangle(ori_prev, (match_upper_left_y,match_upper_left_x), (match_lower_right_y,match_lower_right_x), (0,0,255), 1)
            # cv.rectangle(ori_prev, ((y-half_template_size)*ds_factor,(x-half_template_size)*ds_factor), ((y+half_template_size)*ds_factor,(x+half_template_size)*ds_factor), (255,0,0), 1) 
            cv.rectangle(ori_img, (match_upper_left_x,match_upper_left_y), (match_lower_right_x,match_lower_right_y), (0,0,255), 1)
            cv.rectangle(ori_img, ((x-half_template_size)*ds_factor,(y-half_template_size)*ds_factor), ((x+half_template_size)*ds_factor,(y+half_template_size)*ds_factor), (255,0,0), 1) 
            cv.rectangle(ori_prev, (match_upper_left_x,match_upper_left_y), (match_lower_right_x,match_lower_right_y), (0,0,255), 1)
            cv.rectangle(ori_prev, ((x-half_template_size)*ds_factor,(y-half_template_size)*ds_factor), ((x+half_template_size)*ds_factor,(y+half_template_size)*ds_factor), (255,0,0), 1) 
            
            # ori_img[x*ds_factor,y*ds_factor] = np.array([255,0,0]) 
            # ori_prev[x*ds_factor,y*ds_factor] = np.array([255,0,0]) 

            # ori_img[(x+match_ul_x)*ds_factor,(y+match_ul_y)*ds_factor] = np.array([0,0,255]) 
            # ori_prev[(x+match_ul_x)*ds_factor,(y+match_ul_y)*ds_factor] = np.array([0,0,255])

            
            prev_block_ori = ori_prev[match_upper_left_y:match_lower_right_y,match_upper_left_x:match_lower_right_x]
            next_block_ori = ori_img[(y-half_template_size)*ds_factor:(y+half_template_size)*ds_factor,\
                                        (x-half_template_size)*ds_factor:(x+half_template_size)*ds_factor,:]
            # prev_block_ori = ori_prev[match_upper_left_x:match_lower_right_x,match_upper_left_y:match_lower_right_y]
            # next_block_ori = ori_img[(x-half_template_size)*ds_factor:(x+half_template_size)*ds_factor,\
            #                             (y-half_template_size)*ds_factor:(y+half_template_size)*ds_factor,:]
            prev_block = source[match_ul_y:match_ul_y+w+1, match_ul_x:match_ul_x+w+1]
            next_block = template

            prev_gray_cp =  prev_gray.copy()
            next_gray_cp =  next_gray.copy()

            # cv.rectangle(prev_gray_cp, (prev_match_y-half_template_size, prev_match_x-half_template_size),\
            #                             (prev_match_y+half_template_size, prev_match_x+half_template_size), 255)
            # cv.rectangle(next_gray_cp, (y-half_template_size, x-half_template_size),(y+half_template_size,x+half_template_size), 255)
            cv.rectangle(prev_gray_cp, (prev_match_x-half_template_size, prev_match_y-half_template_size),\
                                        (prev_match_x+half_template_size, prev_match_y+half_template_size), 255)
            cv.rectangle(next_gray_cp, (x-half_template_size, y-half_template_size),(x+half_template_size,y+half_template_size), 255)


            # cv.rectangle(ori_img, (yy_ul*ds_factor, xx_ul*ds_factor), (yy_lr*ds_factor, xx_lr*ds_factor), (0,255,0), 1)
            # cv.rectangle(ori_prev, (yy_ul*ds_factor, xx_ul*ds_factor), (yy_lr*ds_factor, xx_lr*ds_factor), (0,255,0), 1)
            cv.rectangle(ori_img, (xx_ul*ds_factor, yy_ul*ds_factor), (xx_lr*ds_factor, yy_lr*ds_factor), (0,255,0), 1)
            cv.rectangle(ori_prev, (xx_ul*ds_factor, yy_ul*ds_factor), (xx_lr*ds_factor, yy_lr*ds_factor), (0,255,0), 1)


    plt.figure(figsize=(32,15))

    plt.subplot(241)
    plt.imshow(ori_prev)
    plt.subplot(242)
    plt.imshow(ori_img)

    plt.subplot(243)
    plt.imshow(prev_block_ori)
    plt.subplot(244)
    plt.imshow(next_block_ori)


    plt.subplot(245)
    plt.imshow(prev_gray_cp)
    plt.subplot(246)
    plt.imshow(next_gray_cp)

    plt.subplot(247)
    plt.imshow(prev_block)
    plt.subplot(248)
    plt.imshow(next_block)

    plt.tight_layout()

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(ori_prev)
    # plt.subplot(122)
    # plt.imshow(ori_img)

    # plt.show()

    # plt.figure(figsize=(12,8))
    # plt.subplot(121)
    # plt.imshow(next_img)
    # plt.subplot(122)
    # plt.imshow(ori_img)
    # plt.show()
    
    plt.savefig(savefilename)

    # plt.figure()
    # plt.imshow(ori_img)
    # plt.savefig('images/block_matching/downby'+str(int(ds_factor))+'/projected/'+next_file[-8::])
    # plt.show()
    # pdb.set_trace()
    return (x, y, prev_match_x, prev_match_y)

def plot_match(prev_file, next_file, ori_prev, ori_file, ds_factor, savefilename, half_template_size=2, half_source_size=4, method = 'hog'): 

    '''
    Inputs: 
    - prev_file: (possibly downsampled)image file to previous frame
    - next_file: (possibly downsampled) image file to current frame of interest
    - ori_file: the original image file to the current frameof interest
    - ds_factor: downsampling factor
    '''

    prev_img = cv.imread(prev_file)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    # prev_gray = cv.medianBlur(prev_gray,3)
 
    next_img = cv.imread(next_file)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
    # next_gray = cv.medianBlur(next_gray,3)

    ori_img = cv.imread(ori_file)
    ori_prev = cv.imread(ori_prev)

    ## define template by around a peak 
    x, y = int(25*8./ds_factor), int(26*8./ds_factor)

    # pdb.set_trace()

    template = next_gray[y-half_template_size:y+half_template_size+1, x-half_template_size:x+half_template_size+1]
    ## define source around the template
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

    source = prev_gray[yy_ul:yy_lr, xx_ul:xx_lr]
    
    
    (x, y, prev_match_x, prev_match_y) = find_match(source, template, next_img, prev_img, next_gray, prev_gray, \
                                        ori_img, ori_prev, ds_factor, method, half_template_size,\
                                        yy_ul, yy_lr, xx_ul, xx_lr, x,y, savefilename)

    return (x, y, prev_match_x, prev_match_y)


def match_in_region(prev_file, next_file, ori_prev, ori_file, ds_factor, x, y, prev_match_x, prev_match_y, prev_ds_factor, savefilename, half_template_size=2, method = 'hog'):
    
    half_source_size = half_template_size * 2

    if ds_factor > prev_ds_factor: 
        raise ValueError

    
    x,y = int(x * prev_ds_factor/ds_factor), (y * prev_ds_factor/ds_factor)
    prev_match_x,prev_match_y = int(prev_match_x * prev_ds_factor/ds_factor), (prev_match_y * prev_ds_factor/ds_factor)
    
    prev_img = cv.imread(prev_file)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    prev_gray = cv.medianBlur(prev_gray,1)

    next_img = cv.imread(next_file)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
    next_gray = cv.medianBlur(next_gray,1)

    ori_img = cv.imread(ori_file)
    ori_prev = cv.imread(ori_prev)


    template = next_gray[y-half_template_size:y+half_template_size+1, x-half_template_size:x+half_template_size+1]
    w, h = template.shape[::-1] 
    w, h = w-1, h-1
    ## define source around the template
    yy_ul, yy_lr = prev_match_y-half_source_size, prev_match_y+half_source_size+1
    xx_ul, xx_lr =  prev_match_x-half_source_size, prev_match_x+half_source_size+1

    if yy_ul < 0: 
        yy_ul = 0
    if xx_ul < 0: 
        xx_ul = 0
    if yy_lr > prev_gray.shape[0]-1:
        yy_lr = prev_gray.shape[0]-1
    if xx_lr > prev_gray.shape[1]-1: 
        xx_lr = prev_gray.shape[1]-1

    # source = prev_gray[yy_ul:yy_lr, xx_ul:xx_lr]
    source = prev_gray[yy_ul:yy_lr, xx_ul:xx_lr]
    if template.shape[0] == 0:
        pdb.set_trace()
    (x, y, prev_match_x, prev_match_y) = find_match(source, template, next_img, prev_img, next_gray, prev_gray, \
                                        ori_img, ori_prev, ds_factor, method, half_template_size,\
                                        yy_ul, yy_lr, xx_ul, xx_lr, x,y, savefilename)

    return (x, y, prev_match_x, prev_match_y)

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




method = cv.TM_CCORR_NORMED#cv.TM_CCORR #'hog'# cv.TM_CCORR

half_template_size = 4


savefilepath = 'images/demo_3hierarchy_bm/'

path = 'images/downsamplePlot/down_by_8/'
files = glob.glob(path+"*.jpg")
files = sorted(files)
ori_path = 'images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)


# file1 = path+'4063.jpg'
# file2 = path+'4064.jpg'
# ori1 = ori_path+'4063.jpg'
# ori2 = ori_path+'4064.jpg'

f1 = '4067.jpg'
f2 = '4068.jpg'
file1 = path+f1
file2 = path+f2
ori1 = ori_path+f1
ori2 = ori_path+f2


if method == 'hog': 
    savefilename = savefilepath+'HOG/'+'step0_down_by_8.png'
elif method == cv.TM_CCORR:
    savefilename = savefilepath+'TM_CCORR/'+'step0_down_by_8.png'
elif method == cv.TM_CCORR_NORMED:
    savefilename = savefilepath+'TM_CCORR_NORMED/'+'step0_down_by_8.png'

(x, y, prev_match_x, prev_match_y) = plot_match(file1, file2, ori1, ori2, 8, savefilename, half_template_size = half_template_size, half_source_size=6, method=method)
# x, y = 48/2, 52/2
# (x, y, prev_match_x, prev_match_y) = match_in_region(file1, file2, ori1, ori2, 8, x, y, x, y, 8, savefilename, half_template_size = half_template_size, method =method)# 



path = 'images/downsamplePlot/down_by_4/'
files = glob.glob(path+"*.jpg")
files = sorted(files)

f1 = '4067.jpg'
f2 = '4068.jpg'
file1 = path+f1
file2 = path+f2

if method == 'hog': 
    savefilename = savefilepath+'HOG/'+'step1_down_by_4.png'
elif method == cv.TM_CCORR:
    savefilename = savefilepath+'TM_CCORR/'+'step1_down_by_4.png'
elif method == cv.TM_CCORR_NORMED:
    savefilename = savefilepath+'TM_CCORR_NORMED/'+'step1_down_by_4.png'

# (x, y, prev_match_x, prev_match_y) = match_in_region(file1, file2, ori1, ori2, 4, x, y, prev_match_x, prev_match_y,8, savefilename, half_template_size = half_template_size, method =method)# 
# (x, y, prev_match_x, prev_match_y) = plot_match(file1, file2, ori1, ori2, 4, savefilename, half_template_size = half_template_size, method=method)
# x, y = 48, 52
# (x, y, prev_match_x, prev_match_y) = match_in_region(file1, file2, ori1, ori2, 4, x, y, x, y, 4, savefilename, half_template_size = half_template_size, method =method)# 



path = 'images/downsamplePlot/down_by_2/'
files = glob.glob(path+"*.jpg")
files = sorted(files)

f1 = '4067.jpg'
f2 = '4068.jpg'
file1 = path+f1
file2 = path+f2

if method == 'hog': 
    savefilename = savefilepath+'HOG/'+'step2_down_by_2.png'
elif method == cv.TM_CCORR:
    savefilename = savefilepath+'TM_CCORR/'+'step2_down_by_2.png'
elif method == cv.TM_CCORR_NORMED:
    savefilename = savefilepath+'TM_CCORR_NORMED/'+'step2_down_by_2.png'

# (x, y, prev_match_x, prev_match_y) = match_in_region(file1, file2, ori1, ori2, 2, x, y, prev_match_x, prev_match_y,4, savefilename, half_template_size = half_template_size, method =method)# 
(x, y, prev_match_x, prev_match_y) = match_in_region(file1, file2, ori1, ori2, 2, x, y, prev_match_x, prev_match_y,8, savefilename, half_template_size = half_template_size, method =method)# 


if method == 'hog': 
    savefilename = savefilepath+'HOG/'+'step3_ori.png'
elif method == cv.TM_CCORR:
    savefilename = savefilepath+'TM_CCORR/'+'step3_ori.png'
elif method == cv.TM_CCORR_NORMED:
    savefilename = savefilepath+'TM_CCORR_NORMED/'+'step3_ori.png'

(x, y, prev_match_x, prev_match_y) = match_in_region(ori1, ori2, ori1, ori2, 1, x, y, prev_match_x, prev_match_y,2, savefilename, half_template_size = half_template_size, method =method)



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


