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
border = 20
to_remove_outlier= True
def helper_get_normalized_residual(x,y, v_x, v_y, neighbors, velocity): 
    if len(neighbors)> 1: 
        dxy = []
        vel_x = []
        vel_y = []
        for i,pt in enumerate(neighbors): 
            neighbor_x, neighbor_y = pt[0], pt[1]

            dxy=np.sqrt((x-neighbor_x)**2 + (y-neighbor_y)**2)                
            vel_x.append(velocity[i][0])
            vel_y.append(velocity[i][1])
    

        median_d = np.median(np.array(dxy))    
        # median_d_x = np.median(np.array(dx))    
        # median_d_y = np.median(np.array(dy))    

        # epi_x = (-median_d_x+ np.sqrt(median_d_x**2+0.4))/2
        # epi_y = (-median_d_y+ np.sqrt(median_d_y**2+0.4))/2
        epi = (-median_d+ np.sqrt(median_d**2+0.4))/2

        # norm_median_x = np.median(vel_x/(dx+epi_x))
        # norm_median_y = np.median(vel_y/(dy+epi_y))
        norm_median_x = np.median(vel_x/(dxy+epi))
        norm_median_y = np.median(vel_y/(dxy+epi))

        # r_x = abs(v_x/(median_d_x + epi_x) - norm_median_x) / ( np.median(np.abs(vel_x/(dx + epi_x) - norm_median_x)) + epi_x)
        # r_y = abs(v_y/(median_d_y + epi_y) - norm_median_y) / ( np.median(np.abs(vel_y/(dy + epi_y) - norm_median_y)) + epi_y)
        r_x = abs(v_x/(median_d + epi) - norm_median_x) / ( np.median(np.abs(vel_x/(dxy+epi) - norm_median_x)) + epi)
        r_y = abs(v_y/(median_d + epi) - norm_median_y) / ( np.median(np.abs(vel_y/(dxy+epi)- norm_median_y)) + epi)

        # print(r_x, r_y)
        # pdb.set_trace()

        r = np.sqrt(r_x**2 + r_y**2)
    else: 
        return 0

    # pdb.set_trace()
    return r

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
    # if  (y-half_template_size < 0) | (y+half_template_size+1 > prev_gray.shape[0]-1) | \
    #     (x-half_template_size < 0) | (x+half_template_size+1 > prev_gray.shape[1]-1):
    #     # return np.Inf, np.Inf
    #     return x,y
    template_yy_ul, template_yy_lr = max(0,y-half_template_size), min(prev_gray.shape[0],y+half_template_size+1)
    template_xx_ul, template_xx_lr = max(0,x-half_template_size), min(prev_gray.shape[1],x+half_template_size+1)

    template = prev_gaussian[template_yy_ul:template_yy_lr, template_xx_ul:template_xx_lr]

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

    (next_match_x, next_match_y) = block_match(source, template, method, half_template_size, \
                                    yy_ul, yy_lr, xx_ul, xx_lr, x,y,\
                                    template_yy_ul, template_yy_lr, template_xx_ul, template_xx_lr)

    if next_match_x == np.Inf:
        
        # fig, (ax1, ax2) = plt.subplots(1,2,sharex=True, sharey=True)
        # ax1.imshow(template)
        # ax1.set_title('Template')
        # ax2.imshow(source)
        # ax2.set_title('Source')
        # plt.show()
        
        # pdb.set_trace()
        print("no match!")
        return np.Inf, np.Inf

    return next_match_x, next_match_y

def get_tracking_points():
    global with_warp, border
    # '''Get points of interest based on intensity in downsampled by 8 images (for limited number to track)'''
    f = ori_path + '4051.jpg'
    img = cv.imread(f)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if with_warp: 
        gray_warp = cv.remap(gray, warp_x, warp_y, cv.INTER_LINEAR)
        gray_warp = cv.copyMakeBorder(gray_warp, border, border, border, border, cv.BORDER_CONSTANT)
        img2 = cv.pyrDown(gray_warp)
    else:
        img2 = cv.pyrDown(gray)
    img4 = cv.pyrDown(img2)
    gray = cv.pyrDown(img4)
    # gray=img4

    _, thresh = cv.threshold(gray,80,1,cv.THRESH_BINARY)
    coordinates = []
    for y in range(thresh.shape[0]):
        for x in range(thresh.shape[1]):
            if thresh[y,x] > 0:
                coordinates.append([x*8, y*8])
                # coordinates.append([x*4, y*4])
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

def gaussian_blur_bm(make_plots = False, half_template_size = 16, half_source_size = 23):

    global with_warp, blur_window_size, border
    xys, neighbors = get_tracking_points()

    method = cv.TM_CCORR_NORMED#cv.TM_CCOEFF_NORMED #cv.TM_CCORR_NORMED
    
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
    

    xy_list = np.zeros((len(xys), 28, 2))
    xy_list[:,0,:] = xys

    for ii, i in enumerate(range(4051,4072)): #4078
        f1 = str(i)+'.jpg'
        f2 = str(i+1)+'.jpg'
        if make_plots:
            print('----')
            print(f2)

        ori_file = ori_path + f2
        ori_prev = ori_path + f1

        prev_img = cv.imread(ori_prev)
        prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
        next_img = cv.imread(ori_file)
        next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)

        if with_warp:
            prev_gray = cv.remap(prev_gray, warp_x, warp_y, cv.INTER_LINEAR)
            next_gray = cv.remap(next_gray, warp_x, warp_y, cv.INTER_LINEAR)
        
            prev_gray = cv.copyMakeBorder(prev_gray, border, border, border, border, cv.BORDER_CONSTANT)
            prev_img = cv.cvtColor(prev_gray,cv.COLOR_GRAY2RGB)

            next_gray = cv.copyMakeBorder(next_gray, border, border, border, border, cv.BORDER_CONSTANT)
            next_img = cv.cvtColor(next_gray,cv.COLOR_GRAY2RGB)

        if (with_warp==True) & (make_plots==True):
            plt.figure(figsize=(21, 12))
            plt.subplot(131)
            plt.imshow(next_img)
            plt.subplot(132)
            next_gaussian = cv.GaussianBlur(next_gray,(blur_window_size,blur_window_size),0)
            plt.imshow(next_gaussian, cmap='gray')
        elif make_plots:
            plt.figure(figsize=(15,15))

        ## Find match
        for j in range(num_pts):
            x, y = int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])

            next_match_x, next_match_y = find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)

            if next_match_x == np.Inf: 
                # print("TODO: handle no match cases")
                next_match_x, next_match_y = x, y
                # pdb.set_trace()

            x, y = next_match_x, next_match_y
            xy_list[j,ii+1,:] = [x, y] 
            
        # if f2=='4057.jpg':
        #     pdb.set_trace()

        for j in range(num_pts): 
            ## Smooth outlier
            outlier = False
            x, y = xy_list[j,ii+1,0], xy_list[j,ii+1,1]
            if to_remove_outlier: 
                v_x, v_y = (xy_list[j,ii+1,0]-xy_list[j,ii,0])/40., \
                            (xy_list[j,ii+1,1]-xy_list[j,ii,1])/40.
                neighbor_idxs = neighbors[j]
                if len(neighbor_idxs) > 0: 

                    neighbor_xys = []
                    neighbor_vel = []
                    for idx in neighbor_idxs: 
                        neighbor_xys.append(xy_list[idx,ii+1,:])
                        neighbor_vel.append((xy_list[idx,ii+1,:] - xy_list[idx,ii,:])/40.)
                    # if (ii == 3) & (xy_list[j,ii+1,0] > 30) & (xy_list[j,ii+1,1] > 30):
                    #     pdb.set_trace()

                    r = helper_get_normalized_residual(x,y, v_x, v_y, neighbor_xys, neighbor_vel)
                    if r > 1.5: # outlier!
                        # pdb.set_trace()
                        if make_plots:
                            cv.circle(next_img, (int(x), int(y)), 3, (0, 255, 255))
                        
                        vxy = np.array(neighbor_vel).mean(axis=0)
                        xy_list[j,ii+1,:] = vxy*40. + xy_list[j,ii,:]
                        x, y = xy_list[j,ii+1,0], xy_list[j,ii+1,1]
                        # x, y = xy[0], xy[1]
                        # xy_list[j,ii+1,:] = [x, y]
                        print('outlier!')
                        # print(j, [x,y])

                        outlier=True
            # # Plot
            # first_x, first_y = xy_list[j, 0, 0], xy_list[j, 0, 1] 
            # cv.circle(next_img, (int(first_x), int(first_y)), 3, (0, 255, 0))
            if make_plots:
                prev_x, prev_y = xy_list[j, ii, 0], xy_list[j, ii, 1] 
                cv.arrowedLine(next_img, (int(prev_x), int(prev_y)), (int(x), int(y)), (255,0,0), 1, tipLength=0.3)

                if outlier==True:
                    cv.circle(next_img, (int(x), int(y)), 2, (0, 255, 0))
                    cv.arrowedLine(next_img, (int(prev_x), int(prev_y)), (int(x), int(y)), (0,255,0), 1, tipLength=0.3)
                else:
                    cv.circle(next_img, (int(x), int(y)), 2, (255, 255, 0))

            # if f2=='4071.jpg':
            #     cv.circle(next_img, (int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])), 3, (255, 0, 0))
            #     cv.arrowedLine(next_img, (int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])), (int(x), int(y)), (0,255,0), 1, tipLength=0.3)

        # plt.imsave('../images/gaussian_blur/'+f2, next_img)
        if make_plots:
            if with_warp:
                plt.subplot(133)
            plt.imshow(next_img)
            # pdb.set_trace()
            plt.savefig('../images/gaussian_blur/'+f2)



    if make_plots:
        print("compare first and last of cycle")
    # pdb.set_trace()
    f1 = '4052.jpg'
    f2 = '4071.jpg'
    if make_plots:
        ori_prev = ori_path + f1
        ori_file = ori_path + f2
        print(ori_prev)
        print(ori_file)

    prev_img = cv.imread(ori_prev)
    prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
    next_img = cv.imread(ori_file)
    next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)
    if with_warp:
        prev_gray = cv.remap(prev_gray, warp_x, warp_y, cv.INTER_LINEAR)
        next_gray = cv.remap(next_gray, warp_x, warp_y, cv.INTER_LINEAR)
    
        prev_gray = cv.copyMakeBorder(prev_gray, border, border, border, border, cv.BORDER_CONSTANT)
        prev_img = cv.cvtColor(prev_gray,cv.COLOR_GRAY2RGB)

        next_gray = cv.copyMakeBorder(next_gray, border, border, border, border, cv.BORDER_CONSTANT)
        next_img = cv.cvtColor(next_gray,cv.COLOR_GRAY2RGB)
    
    plt.figure(figsize=(14,12))
    plt.subplot(121)
    plt.imshow(next_img)

    distances = []

    for j in range(num_pts):

        x, y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])
        next_match_x, next_match_y = find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)

        if next_match_x != np.Inf: 
            
            if make_plots:
                prev_x, prev_y = xy_list[j, 0, 0], xy_list[j, 0, 1] 
                cv.arrowedLine(next_img, (int(prev_x), int(prev_y)), \
                            (int(next_match_x), int(next_match_y)), (255,0,0), 1, tipLength=0.3)

                cv.circle(next_img, (int(next_match_x), int(next_match_y)), 1, (255, 255, 0))
            
                cv.circle(next_img, (int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])), 1, (255, 0, 0))
                cv.arrowedLine(next_img, (int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])), (int(next_match_x), int(next_match_y)), (0,255,0), 1, tipLength=0.3)
            
            distances.append(np.sqrt((int(xy_list[j, ii, 0])- next_match_x)**2 + \
                                (int(xy_list[j, ii, 1])-next_match_y)**2))

    # plt.imsave('../images/gaussian_blur/4052_4071.jpg', next_img)
    if make_plots:
        plt.subplot(122)
        plt.imshow(next_img)
        plt.savefig('../images/gaussian_blur/4052_4071.jpg')

    distances = np.array(distances)
    file  = open('../images/gaussian_blur/error.txt', 'a')
    file.write('Distances between tracked location and direct step location in pixel:\n')
    file.write('half template size: %.1f, half source size: %.1f\n' % (half_template_size, half_source_size))
    file.write('Mean: %.4f\n' % distances.mean())
    file.write('Min: %.4f\n' % distances.min())
    file.write('Max: %.4f\n' % distances.max())
    file.write('Standard Deviation: %.4f\n' % distances.std())
    return distances.mean(), distances.std()

if __name__ == "__main__":
    # execute only if run as a script

    # res = np.full((26, 7, 2), np.inf)

    # half_temp_sizes = np.arange(15, 41)
    # for idxi, i in enumerate(half_temp_sizes): 
    #     for idxj, j in enumerate(np.arange(5,12)):
    #         half_template_size = i
    #         half_source_size = i+j
    #         # print(half_template_size, half_source_size)
    #         mean, std = gaussian_blur_bm(half_template_size=half_template_size, half_source_size=half_source_size)
    #         res[idxi, idxj, :] = [mean, std]
    #         print(half_template_size, half_source_size, mean, std)
    
    # np.save('../images/gaussian_blur/gridsearch_result_full2.npy', res)
    # # pdb.set_trace()

    # min_mean = res[:,:,0].min()
    # min_std = res[:,:,1].min()

    # min_mean_loc = np.where(res == min_mean)
    # print('min mean at half_temp_size=%.1f, half source size=%.1f' % \
    #                     (half_temp_sizes[min_mean_loc[0][0]], \
    #                         half_temp_sizes[min_mean_loc[0][0]]+np.arange(5,11)[min_mean_loc[1][0]]))

    # min_std_loc = np.where(res == min_std)
    # print('min std at half_temp_size=%.1f, half source size=%.1f' % \
    #                     (half_temp_sizes[min_std_loc[0][0]], \
    #                         half_temp_sizes[min_std_loc[0][0]]+np.arange(5,11)[min_std_loc[1][0]]))


    # plt.figure()
    # plt.imshow(res)
    # plt.show()

    # res = np.load('../images/gaussian_blur/gridsearch_result_full2.npy')
    # pdb.set_trace()
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(res[:,:,0])
    # plt.title('mean')
    # plt.subplot(122)
    # plt.imshow(res[:,:,1])
    # plt.title('std')
    
    # plt.show()

    # pdb.set_trace()


    half_template_size = 30
    half_source_size = 38
    mean, std = gaussian_blur_bm(half_template_size=half_template_size, half_source_size=half_source_size, make_plots=True)


# (Pdb) min_mean = res[:,:,0].min()
# (Pdb) min_mean
# 4.361738954332011

# (Pdb) min_std
# 3.0604626020052077

# (Pdb) np.where(res == min_mean)
# (array([20]), array([4]), array([0]))
# (Pdb) np.where(res == min_std)
# (array([20]), array([4]), array([1]))

#--> half_template_size = 25, half_source_size = 34