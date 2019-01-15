''' Block matching to the neighboring pixels'''

import cv2 as cv
import numpy as np
import pdb
import glob
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

from hog import get_hog


def block_matching(template_gray, source_gray,  method=cv.TM_CCORR): 
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
    # pdb.set_trace()
    
    # # Draw a rectangle around the matched region. 
    # w, h = template_gray.shape[::-1] 
    # for pt in zip(*loc[::-1]): 
    #     print(pt)
    #     cv.rectangle(source_gray, pt, (pt[0] + w, pt[1] + h), 255, 1) 

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(template_gray)
    # plt.subplot(122)
    # plt.imshow(source_gray)
    # plt.show()

    return loc

def plot_match(prev_file, next_file, ori_prev, ori_file, ds_factor, half_template_size=2, half_source_size=4): 

    '''
    Inputs: 
    - prev_file: (possibly downsampled)image file to previous frame
    - next_file: (possibly downsampled) image file to current frame of interest
    - ori_file: the original image file to the current frameof interest
    - ds_factor: downsampling factor
    '''

    got_one = False

    prev_img = cv.imread(prev_file)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    # prev_gray = cv.medianBlur(prev_gray,3)
 
    next_img = cv.imread(next_file)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
    # next_gray = cv.medianBlur(next_gray,3)

    ori_img = cv.imread(ori_file)
    ori_prev = cv.imread(ori_prev)

    # coordinates = peak_local_max(next_gray, threshold_abs = 100, min_distance=3) # footprint=np.ones((5,5)) for original
    _, thresh = cv.threshold(next_gray,100,1,cv.THRESH_BINARY)
    coordinates = []
    for x in range(thresh.shape[0]):
        for y in range(thresh.shape[1]):
            if thresh[x,y] > 0:
                coordinates.append([x,y])
    
    # pdb.set_trace()
    
    ## define template by around a peak 
    for xy in coordinates:
        # xy = coordinates[0]
        try:
            x, y = xy[0], xy[1]
        except:
            print(xy)
            pdb.set_trace()

        template = next_gray[y-half_template_size:y+half_template_size+1, x-half_template_size:x+half_template_size+1]
        w, h = template.shape[::-1] 
        w, h = w-1, h-1
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
        
        loc = block_matching(template, source)
        next_img[x,y] = np.array([0,0,255]) 

        cv.circle(ori_img, (y*ds_factor, x*ds_factor), 1, (255,0,0),1)

        if len(loc)>0:
            # cv.circle(next_img, (y, x), 1, (255, 0, 0))
            # cv.rectangle(ori, (y-half_source_size, x-half_source_size),(y+half_source_size, x+half_source_size), (0,255,0), 1) 
            # cv.rectangle(next_img, (y-half_template_size, x-half_template_size),(y+half_template_size, x+half_template_size), (0,255,0), 1)    
            # print('new', (y-half_template_size, x-half_template_size),(y+half_template_size, x+half_template_size))

            # # Draw a rectangle around the matched region. 
            for pt in zip(*loc[::-1]): 
                # cv.rectangle(prev_img, (y-half_source_size+pt[0], x-half_source_size+pt[1]), (y-half_source_size+pt[0] + w, x-half_source_size+pt[1] + h), (0,0,255), 1)
                # prev_img[y-half_source_size+pt[0]+half_template_size, x-half_source_size+pt[1]+half_template_size] = np.array([0,0,255])
                prev_img[x-half_source_size+pt[1]+half_template_size, y-half_source_size+pt[0]+half_template_size] = np.array([0,0,255])
                
                # print('prev', (y-half_source_size+pt[0], x-half_source_size+pt[1]), (y-half_source_size+pt[0] + w, x-half_source_size+pt[1] + h)) 
                # cv.arrowedLine(next_img, (y-half_source_size+pt[0]+half_template_size+1, x-half_source_size+pt[1]+half_template_size+1), (y,x), (255,255,0), 1, tipLength=0.3)
                
                prev_match_y = y-half_source_size+pt[0]+half_template_size
                prev_match_x = x-half_source_size+pt[1]+half_template_size
                
                cv.line(next_img, (prev_match_y, prev_match_x), (y,x), (255,255,0), 1)

                ## TODO Uncomment the following line
                cv.arrowedLine(ori_img, (prev_match_y*ds_factor, prev_match_x*ds_factor), \
                        (y*ds_factor,x*ds_factor), (255,255,0), 1, tipLength=0.3)

                # cv.rectangle(next_img, (y-half_source_size+pt[0], x-half_source_size+pt[1]), (y-half_source_size+pt[0] + w, x-half_source_size+pt[1] + h), (0,0,255), 1) 

                # print((y-half_template_size, x-half_template_size),(y+half_template_size, x+half_template_size))
                # print((y-half_source_size+pt[0], x-half_source_size+pt[1]), (y-half_source_size+pt[0] + w, x-half_source_size+pt[1] + h))
                next_img[x,y] = np.array([255,0,0]) 

                if (y==26) & (got_one==False):
                    got_one = True
                    cv.circle(ori_img, (y*ds_factor, x*ds_factor), 2, (255, 0, 0))
                    match_upper_left_y = (prev_match_y-half_template_size)*ds_factor
                    match_upper_left_x = (prev_match_x-half_template_size)*ds_factor
                    
                    match_lower_right_y = (prev_match_y+half_template_size)*ds_factor
                    match_lower_right_x = (prev_match_x+half_template_size)*ds_factor
                    
                    cv.rectangle(ori_img, (match_upper_left_y,match_upper_left_x), (match_lower_right_y,match_lower_right_x), (0,0,255), 1)
                    cv.rectangle(ori_img, ((y-half_template_size)*ds_factor,(x-half_template_size)*ds_factor), ((y+half_template_size)*ds_factor,(x+half_template_size)*ds_factor), (255,0,0), 1) 
                    # cv.rectangle(prev_img, (y-half_source_size+pt[0], x-half_source_size+pt[1]), (y-half_source_size+pt[0] + w, x-half_source_size+pt[1] + h), (0,0,255), 1)
                    cv.rectangle(ori_prev, (match_upper_left_y,match_upper_left_x), (match_lower_right_y,match_lower_right_x), (0,0,255), 1)
                    cv.rectangle(ori_prev, ((y-half_template_size)*ds_factor,(x-half_template_size)*ds_factor), ((y+half_template_size)*ds_factor,(x+half_template_size)*ds_factor), (255,0,0), 1) 
                    
                    ori_img[x*ds_factor,y*ds_factor] = np.array([255,0,0]) 
                    ori_prev[x*ds_factor,y*ds_factor] = np.array([255,0,0]) 

                    ori_img[(x+pt[1])*ds_factor,(y+pt[0])*ds_factor] = np.array([0,0,255]) 
                    ori_prev[(x+pt[1])*ds_factor,(y+pt[0])*ds_factor] = np.array([0,0,255])

                    cv.line(ori_img, (prev_match_y*ds_factor, prev_match_x*ds_factor), (y*ds_factor,x*ds_factor), (255,255,0), 1) 
                    cv.line(ori_prev, (prev_match_y*ds_factor, prev_match_x*ds_factor), (y*ds_factor,x*ds_factor), (255,255,0), 1) 

                    prev_block_ori = ori_prev[match_upper_left_x:match_lower_right_x,match_upper_left_y:match_lower_right_y]
                    next_block_ori = ori_img[(x-half_template_size)*ds_factor:(x+half_template_size)*ds_factor,\
                                            (y-half_template_size)*ds_factor:(y+half_template_size)*ds_factor,:]
                    prev_block = source[pt[1]:pt[1]+w+1,pt[0]:pt[0]+w+1]
                    next_block = template

                    prev_gray_cp =  prev_gray.copy()
                    next_gray_cp =  next_gray.copy()

                    cv.rectangle(prev_gray_cp, (y-half_template_size+pt[0], x-half_template_size+pt[1]),\
                                            (y+half_template_size+pt[0],x+half_template_size+pt[1]), 255)
                    cv.rectangle(next_gray_cp, (y-half_template_size, x-half_template_size),(y+half_template_size,x+half_template_size), 255)


                    cv.rectangle(ori_img, (yy_ul*ds_factor, xx_ul*ds_factor), (yy_lr*ds_factor, xx_lr*ds_factor), (0,255,0), 1)
                    cv.rectangle(ori_prev, (yy_ul*ds_factor, xx_ul*ds_factor), (yy_lr*ds_factor, xx_lr*ds_factor), (0,255,0), 1)

                    # pdb.set_trace()

        # else: 
        #     print("No match!")
                

    plt.figure(figsize=(25,12))

    plt.subplot(241)
    plt.imshow(ori_prev)
    plt.subplot(242)
    plt.imshow(ori_img)

    plt.subplot(243)
    plt.imshow(prev_block_ori)
    plt.subplot(244)
    plt.imshow(next_block_ori)

    plt.subplot(245)
    plt.imshow(prev_block)
    plt.subplot(246)
    plt.imshow(next_block)

    plt.subplot(247)
    plt.imshow(prev_gray_cp)
    plt.subplot(248)
    plt.imshow(next_gray_cp)

    plt.tight_layout()

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(ori_prev)
    # plt.subplot(122)
    # plt.imshow(ori_img)

    plt.show()

    # plt.figure(figsize=(12,8))
    # plt.subplot(121)
    # plt.imshow(next_img)
    # plt.subplot(122)
    # plt.imshow(ori_img)
    # plt.show()
    
    # plt.savefig('images/block_matching/downby'+str(int(ds_factor))+'/'+next_file[-8::])

    # plt.figure()
    # plt.imshow(ori_img)
    # plt.savefig('images/block_matching/downby'+str(int(ds_factor))+'/projected/'+next_file[-8::])
    # plt.show()
    # pdb.set_trace()





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

plot_match(file1, file2, ori1, ori2, 8, half_template_size=7, half_source_size=10)

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


