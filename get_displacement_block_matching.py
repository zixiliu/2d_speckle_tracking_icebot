import cv2 as cv
from python_hierachy_block_matching.block_matching import block_match
from python_hierachy_block_matching.trace_trajectory import find_match
import pdb
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max

################################################################################
## Global variables
################################################################################
# ori_path = 'images/original/'
ori_path = '/Users/zixiliu/my_git_repos/my_howe_lab/Echos/Zeo file/IM_1637_copy_jpg/'
# ori_path = '/Users/zixiliu/my_git_repos/my_howe_lab/Echos/Zeo file/images/'
# ori_path = 'images/simulated/'
# save_img_path = 'images/block_matching/displacement/'
save_img_path = ori_path+'tracking/'

# save_path = 'images/block_matching/'
save_path = save_img_path

# ori_path='validation_simulation/'
# ori_path = '/Users/zixiliu/my_git_repos/2d_speckle_tracking_icebot/images/warped/'
# the_ori_path = '/Users/zixiliu/my_git_repos/2d_speckle_tracking_icebot/images/original/'
the_ori_path = ori_path
# the_ori_path = '/Users/zixiliu/my_git_repos/2d_speckle_tracking_icebot/images/simulated/'
# save_img_path = ori_path + 'displacement/'
# save_path = ori_path

ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)
with_warp = False
blur_window_size = 9
border = 20
to_remove_outlier= True
plot_center_pt = False

# starting_img_num = 4065 #4051
# ending_img_num = 4084 #4072
# starting_img_num = 0
# ending_img_num = 19
# starting_img_num = 4050
# ending_img_num = 4069
# starting_img_num = 4060
# ending_img_num = 4079
starting_img_num = 1
ending_img_num = 34



dense_track_grid = False

if with_warp:
    warp_x = np.load('python_hierachy_block_matching/warpx.npy')
    warp_y = np.load('python_hierachy_block_matching/warpy.npy')

use_groundtruth = False
if use_groundtruth:

    ice_path = '/Users/zixiliu/my_git_repos/ICE_segmentation/'
    groundtruth_path = '/Users/zixiliu/my_git_repos/ICE_segmentation/groundtruth/'
    endo = np.load(ice_path+'endo_pts.npy') # (nframe, npoints, (x,y))
    epi = np.load(ice_path+'epi_pts.npy') # (nframe, npoints, (x,y))
    nframe, endo_npoints, _ = endo.shape
    _, epi_npoints, _ = epi.shape

    endo_epi = np.zeros((nframe, endo_npoints+epi_npoints, 2))
    endo_epi[:,0:endo_npoints,:] = endo
    endo_epi[:,endo_npoints::,:] = epi
    # endo_epi = endo

    nframe, npoints, _ = endo_epi.shape
    the_ori_files = glob.glob(ice_path+"original/*.jpg")
    the_ori_files = sorted(the_ori_files)
    groundtruth_files = glob.glob(groundtruth_path+"*.jpg")
    groundtruth_files = sorted(groundtruth_files)


################################################################################
## helper functions
################################################################################
''' Calculate the normalized residual by neighbor points' displacement velocity.
    Helper function for outlier removal. '''
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
        epi = (-median_d+ np.sqrt(median_d**2+0.4))/2
        norm_median_x = np.median(vel_x/(dxy+epi))
        norm_median_y = np.median(vel_y/(dxy+epi))
        r_x = abs(v_x/(median_d + epi) - norm_median_x) / ( np.median(np.abs(vel_x/(dxy+epi) - norm_median_x)) + epi)
        r_y = abs(v_y/(median_d + epi) - norm_median_y) / ( np.median(np.abs(vel_y/(dxy+epi)- norm_median_y)) + epi)
        r = np.sqrt(r_x**2 + r_y**2)
    else:
        return 0
    return r

'''Given (x, y), find match in previous frame.'''
def helper_find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method):
    global blur_window_size
    ## Gaussian Blur
    prev_gaussian = cv.GaussianBlur(prev_gray,(blur_window_size,blur_window_size),0)
    next_gaussian = cv.GaussianBlur(next_gray,(blur_window_size,blur_window_size),0)
    ## Define template by boundary points
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
    ## Block match
    (next_match_x, next_match_y) = block_match(source, template, method, half_template_size, \
                                    yy_ul, yy_lr, xx_ul, xx_lr, x,y,\
                                    template_yy_ul, template_yy_lr, template_xx_ul, template_xx_lr)
    if next_match_x == -1:
        (next_match_x, next_match_y) = x, y
    if next_match_x == np.Inf:
        # print("no match!")
        return np.Inf, np.Inf
    return next_match_x, next_match_y

''' Define the points to track by pixels with high intensity in the downsampled initial frame.
    NOTE: contains magic numbers that encodes the downsample ratio that must be changed accordingly.'''
def helper_get_tracking_points(f):
    global with_warp, border

    # if use_groundtruth==False:
    #     downby = 16 #8 # 4
    #     downby_init = downby
    #     if downby % 2 != 0:
    #         print("Variable downby must be a multiply of 2!")
    #         raise ValueError

    #     img = cv.imread(f)
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     if with_warp:
    #         gray_warp = cv.remap(gray, warp_x, warp_y, cv.INTER_LINEAR)
    #         gray_warp = cv.copyMakeBorder(gray_warp, border, border, border, border, cv.BORDER_CONSTANT)
    #         img2 = cv.pyrDown(gray_warp)
    #     else:
    #         img2 = cv.pyrDown(gray)

    #     while downby > 2:
    #         downby = downby / 2
    #         img2 = cv.pyrDown(img2)

    #     img2Gauss = cv.GaussianBlur(img2,(11,11),0)
    #     _, thresh = cv.threshold(img2Gauss,50,1,cv.THRESH_BINARY)
    #     # _, thresh = cv.threshold(img2,60,1,cv.THRESH_BINARY)



    #     # plt.figure()
    #     # plt.subplot(131)
    #     # plt.imshow(img2, cmap='gray')
    #     # plt.subplot(132)
    #     # temp = cv.GaussianBlur(img2,(11,11),0)
    #     # plt.imshow(temp, cmap='gray')
    #     # plt.subplot(133)
    #     # _, thresh2 = cv.threshold(temp,50,1,cv.THRESH_BINARY)
    #     # plt.imshow(thresh2, cmap='gray')
    #     # plt.show()
    #     # pdb.set_trace()

    #     if dense_track_grid:
    #         coordinates = []
    #         for y in range(3, thresh.shape[0]-3+1):
    #             for x in range(3, thresh.shape[1]-3+1):
    #                     coordinates.append([x*downby_init, y*downby_init])
    #     else:
    #         coordinates = []
    #         for y in range(thresh.shape[0]):
    #             for x in range(thresh.shape[1]):
    #                 if thresh[y,x] > 0:
    #                     coordinates.append([x*downby_init, y*downby_init])


    #     neighbors = {} # Get neighbors by index
    #     # d_sq = 128 # distance thresh
    #     d_sq = 1600 # distance thresh
    #     for i, xy in enumerate(coordinates):
    #         x, y = xy[0], xy[1]
    #         neighbors[i] = []
    #         for j in range(len(coordinates)):
    #             next_xy = coordinates[j]
    #             next_x, next_y = next_xy[0], next_xy[1]
    #             if (x-next_x)**2 + (y-next_y)**2 <= d_sq:
    #                 if ((x==next_x) & (y==next_y)):
    #                     pass
    #                 else:
    #                     neighbors[i].append(j)
    # else:
    #     ice_path = '/Users/zixiliu/my_git_repos/ICE_segmentation/'
    #     global endo_epi
    #     nframe, npoints, _ = endo_epi.shape
    #     coordinates = endo_epi[0,:,:]
    #     # coordinates = np.load(ice_path+'track_pts.npy')
    #     print(coordinates)
    #     # pdb.set_trace()
    #     neighbors = {}
    ice_path = '/Users/zixiliu/my_git_repos/ICE_segmentation/'
    # coordinates = np.load(ice_path+'track_pts.npy')
    # coordinates = [[503, 414],[535, 528],[557, 639],[628, 725],\
    #     [745, 745],[812, 652],[823, 533],[818, 416],[775, 297]]
    # coordinates = [[335, 276],[356, 352],[371, 426],[418, 483],\
    #     [496, 496],[541, 434],[548, 355],[545, 277],[516, 198]]
    # coordinates = [[428, 280], [469, 453], [544, 593], [691, 626], [771, 496], [776, 316], [722, 152]]
    coordinates = [[334, 279], [357, 393], [408, 473], [498, 493], [547, 414], [546, 310], [517, 201]]

    neighbors = {}

    # coordinates = []
    # coordinates.append([70, 90])
    # neighbors = {}
    print("Finished getting coordinates.")
    # pdb.set_trace()
    return coordinates, neighbors

################################################################################
## Global functions
################################################################################
def gaussian_blur_bm(make_plots = False, half_template_size = 16, half_source_size = 23):
    global with_warp, blur_window_size, border
    xys, neighbors = helper_get_tracking_points(ori_path + str(starting_img_num) + '.jpg')
    method = cv.TM_CCORR_NORMED#cv.TM_CCOEFF_NORMED #cv.TM_CCORR_NORMED

    num_pts = len(xys)
    for i, xy in enumerate(xys):
        x, y = xy[0], xy[1]
        # x, y = x*640/1200, y*640/1200
        xys[i] = [x, y]

    num_frames = ending_img_num - starting_img_num + 1
    xy_list = np.zeros((len(xys), num_frames, 2))
    xy_list[:,0,:] = xys
    center_pt = np.zeros((num_frames, 2))
    # pdb.set_trace()

    for ii, i in enumerate(range(starting_img_num,ending_img_num)):
        # if i < 9:
        #     f1 = '0'+str(i)+'.jpg'
        #     f2 = '0'+str(i+1)+'.jpg'
        # else:
        f1 = str(i)+'.jpg'
        f2 = str(i+1)+'.jpg'
        print('----\n'+f2)


        ori_file = ori_path + f2
        ori_prev = the_ori_path + f1

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
            if dense_track_grid:
                x, y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])
            else:
                x, y = int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])
                # x, y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])
            # x, y = int(endo[ii,j,0]), int(endo[ii,j,1])
            # pdb.set_trace()
            next_match_x, next_match_y = helper_find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)

            if next_match_x == np.Inf:
                # print("TODO: handle no match cases")
                next_match_x, next_match_y = x, y

            # x, y = next_match_x, next_match_y
            # xy_list[j,ii+1,:] = [x, y]
            xy_list[j,ii+1,:] = [next_match_x, next_match_y]


        ## Warpped example
        if use_groundtruth:
            groundtruth_img = cv.imread(groundtruth_files[ii+1])
        # groundtruth_img2 = groundtruth_img.copy()
        ## END warpped example

        if ii == 0:
            for j in range(num_pts):
                x, y = xy_list[j,0,0], xy_list[j,0,1]
                cv.circle(prev_img, (int(x), int(y)), 4, (255, 0, 0))
            plt.imsave(save_img_path+f1, prev_img)

        for j in range(num_pts):
            ## Smooth outlier
            outlier = False
            x, y = xy_list[j,ii+1,0], xy_list[j,ii+1,1]
            if to_remove_outlier:
                v_x, v_y = (xy_list[j,ii+1,0]-xy_list[j,ii,0])/40., \
                            (xy_list[j,ii+1,1]-xy_list[j,ii,1])/40.
                try:
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
                            if make_plots:
                                cv.circle(next_img, (int(x), int(y)), 3, (0, 255, 255))

                            vxy = np.array(neighbor_vel).mean(axis=0)
                            xy_list[j,ii+1,:] = vxy*40. + xy_list[j,ii,:]
                            x, y = xy_list[j,ii+1,0], xy_list[j,ii+1,1]
                            # x, y = xy[0], xy[1]
                            # xy_list[j,ii+1,:] = [x, y]
                            # print('outlier!')
                            # print(j, [x,y])

                            outlier=True
                except KeyError:
                    pass
            # # Plot
            # first_x, first_y = xy_list[j, 0, 0], xy_list[j, 0, 1]
            # cv.circle(next_img, (int(first_x), int(first_y)), 3, (0, 255, 0))
            if make_plots:
                # prev_x, prev_y = int(endo_epi[ii,j,0]), int(endo_epi[ii,j,1]) #
                if dense_track_grid:
                    prev_x, prev_y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])
                    # print([x,y])
                else:
                    prev_x, prev_y = int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])
                    # prev_x, prev_y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])

                cv.circle(prev_img, (int(prev_x), int(prev_y)), 3, (0, 255, 0))
                try:
                    cv.arrowedLine(next_img, (int(prev_x), int(prev_y)), (int(x), int(y)), (255,255,0), 1, tipLength=0.3)
                except:
                    print(x, y)
                    pdb.set_trace()
                if use_groundtruth:
                    cv.arrowedLine(groundtruth_img, (int(prev_x), int(prev_y)), (int(x), int(y)), (255,0,0), 2, tipLength=0.3)
                    cv.circle(groundtruth_img, (int(endo_epi[ii+1,j,0]), int(endo_epi[ii+1,j,1])), 1, (0, 255, 0))
                    cv.arrowedLine(next_img, (int(endo_epi[ii,j,0]), int(endo_epi[ii,j,1])), (int(endo_epi[ii+1,j,0]), int(endo_epi[ii+1,j,1])), (0,255,0), 2, tipLength=0.3)
                    # cv.arrowedLine(groundtruth_img, (int(endo_epi[ii,j,0]), int(endo_epi[ii,j,1])), (int(endo_epi[ii+1,j,0]), int(endo_epi[ii+1,j,1])), (0,255,0), 1, tipLength=0.3)

                if outlier==True:
                    cv.circle(next_img, (int(x), int(y)), 2, (0, 255, 0))
                    cv.arrowedLine(next_img, (int(prev_x), int(prev_y)), (int(x), int(y)), (0,255,0), 1, tipLength=0.3)
                else:
                    cv.circle(next_img, (int(x), int(y)), 4, (255, 0, 0))
                    # cv.circle(next_img, (prev_x, prev_y), 3, (255, 0, 0))
                    # if (abs(x-prev_x) != 0) or (abs(y-prev_y) != 0):
                    #     cv.circle(next_img, (int(x), int(y)), 3, (0, 255, 0))
                    if use_groundtruth:
                        cv.circle(groundtruth_img, (int(x), int(y)), 3, (255, 255, 0))

        if plot_center_pt:
            center_pt[ii, :] = [next_img.shape[0]/2+14, next_img.shape[1]/2-14]
            cv.circle(next_img, (next_img.shape[0]/2+14, next_img.shape[1]/2-14), 4, (0, 255, 0))

        # ## plot center points
        # cv.circle(next_img, (int(xy_list[93,ii+1,0]), int(xy_list[93,ii+1,1])), 5, (0, 255, 255))
        # cv.circle(next_img, (int(xy_list[170,ii+1,0]), int(xy_list[170,ii+1,1])), 5, (0, 255, 255))
        # cv.circle(next_img, (int(xy_list[171,ii+1,0]), int(xy_list[171,ii+1,1])), 5, (0, 255, 255))
        # cv.circle(next_img, (int(xy_list[360,ii+1,0]), int(xy_list[360,ii+1,1])), 5, (0, 255, 255))

        # cv.circle(next_img, (int((xy_list[93,ii+1,0]+xy_list[170,ii+1,0]+xy_list[171,ii+1,0]+xy_list[360,ii+1,0])/4), \
        #             int((xy_list[93,ii+1,1]+xy_list[170,ii+1,1]+xy_list[171,ii+1,1]+xy_list[360,ii+1,1])/4)), 5, (0, 255, 0))
        # center_pt[ii, :] = [int((xy_list[93,ii+1,0]+xy_list[170,ii+1,0]+xy_list[171,ii+1,0]+xy_list[360,ii+1,0])/4), \
        #             int((xy_list[93,ii+1,1]+xy_list[170,ii+1,1]+xy_list[171,ii+1,1]+xy_list[360,ii+1,1])/4)]

        ## thresh=80:
        # cv.circle(next_img, (int(xy_list[51,ii+1,0]), int(xy_list[51,ii+1,1])), 5, (0, 255, 255))
        # cv.circle(next_img, (int(xy_list[89,ii+1,0]), int(xy_list[89,ii+1,1])), 5, (0, 255, 255))
        # cv.circle(next_img, (int(xy_list[90,ii+1,0]), int(xy_list[90,ii+1,1])), 5, (0, 255, 255))
        # cv.circle(next_img, (int(xy_list[229,ii+1,0]), int(xy_list[229,ii+1,1])), 5, (0, 255, 255))

        # cv.circle(next_img, (int((xy_list[51,ii+1,0]+xy_list[89,ii+1,0]+xy_list[90,ii+1,0]+xy_list[229,ii+1,0])/4), \
        #             int((xy_list[51,ii+1,1]+xy_list[89,ii+1,1]+xy_list[90,ii+1,1]+xy_list[229,ii+1,1])/4)), 5, (0, 255, 0))
        # cv.circle(next_img, (int((xy_list[89,ii+1,0]+xy_list[131,ii+1,0]+xy_list[132,ii+1,0]+xy_list[278,ii+1,0])/4), \
        #             int((xy_list[89,ii+1,1]+xy_list[131,ii+1,1]+xy_list[132,ii+1,1]+xy_list[278,ii+1,1])/4)), 5, (0, 255, 0))


        # center_pt[ii, :] = [int((xy_list[51,ii+1,0]+xy_list[89,ii+1,0]+xy_list[90,ii+1,0]+xy_list[229,ii+1,0])/4), \
        #             int((xy_list[51,ii+1,1]+xy_list[89,ii+1,1]+xy_list[90,ii+1,1]+xy_list[229,ii+1,1])/4)]

        if make_plots:
            if with_warp:
                plt.subplot(133)
            # cv.rectangle(next_img, (0,0), (3,10), (0,255,0),1)

            # for temp in range(int(xy_list.shape[0]/2)):
            #     cv.line(next_img, (int(xy_list[temp,ii,0]), int(xy_list[temp,ii,1])), (int(xy_list[-1*temp-1,ii,0]), int(xy_list[-1*temp-1,ii,1])), (255,0,0))

            mid_pt = (int((xy_list[0,ii,0] + xy_list[-1,ii,0])/2), int((xy_list[0,ii,1] + xy_list[-1,ii,1])/2))
            # cv.line(next_img, mid_pt, (int(xy_list[4,ii,0]), int(xy_list[4,ii,1])), (255,0,0))
            cv.line(next_img, mid_pt, (int(xy_list[3,ii+1,0]), int(xy_list[3,ii+1,1])), (255,0,0))
            plt.imshow(next_img)

            plt.tight_layout()
            # plt.savefig(save_img_path+f2)
            plt.imsave(save_img_path+f2, next_img)
            ## warp example

            if use_groundtruth:
                plt.figure(figsize=(25,16))
                # plt.subplot(121)
                # plt.imshow(prev_img)
                # plt.title('initial points in the previous image (source)')

                # plt.subplot(122)
                # plt.imshow(next_img)
                # plt.title('Block Matching result on the warpped image')

                # plt.subplot(122)
                plt.imshow(groundtruth_img)
                plt.title('Yellow: Currently Tracked; Green: Ground Truth')
                # plt.title('How block matching results would look on the original(not warpped) image')

                plt.tight_layout()
                plt.savefig(save_img_path+f2)
            # pdb.set_trace()
            plt.close()




    if make_plots:
        print("compare first and last of cycle")
    f1 = str(starting_img_num) + '.jpg'
    f2 = str(ending_img_num) + '.jpg'
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

    # plt.figure(figsize=(14,12))
    plt.figure()
    # plt.subplot(121)
    # plt.imshow(next_img)


    distances = []

    for j in range(num_pts):

        x, y = int(xy_list[j, 0, 0]), int(xy_list[j, 0, 1])
        next_match_x, next_match_y = helper_find_match_gaussian_blur(prev_gray, next_gray, x, y, half_template_size, half_source_size, method)

        if next_match_x != np.Inf:

            if make_plots:
                # cv.arrowedLine(next_img, (x, y), \
                            # (int(next_match_x), int(next_match_y)), (255,255,0), 1, tipLength=0.3)

                # cv.circle(next_img, (int(next_match_x), int(next_match_y)), 1, (255, 255, 0))
                cv.circle(next_img, (x, y), 3, (255, 255, 0))
                cv.circle(next_img, (int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])), 3, (255, 0, 0))
                cv.arrowedLine(next_img, (x,y), (int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])), (0,255,0), 1, tipLength=0.3)
                # cv.arrowedLine(next_img, (int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])), (int(next_match_x), int(next_match_y)), (0,255,0), 1, tipLength=0.3)

                # pdb.set_trace()
                # for temp in range(13):
                #     cv.arrowedLine(next_img, (x[temp+6],y[temp+6]), (), (0,255,0), 1, tipLength=0.3)

            distances.append(np.sqrt((int(xy_list[j, ii, 0])- next_match_x)**2 + \
                                (int(xy_list[j, ii, 1])-next_match_y)**2))


    if make_plots:
        # plt.subplot(122)
        # plt.imshow(next_img)
        # plt.title('Drift \n green circles: tracked points after a cardiac cycle\n yellow cirlces: points to track \n arrow: points after a cycle pointing to direct track')
        # plt.savefig(save_img_path+str(starting_img_num)+'_'+str(ending_img_num)+'.jpg')
        plt.imsave(save_img_path+str(starting_img_num)+'_'+str(ending_img_num)+'.jpg', next_img)
    distances = np.array(distances)
    file  = open(save_path + 'error.txt', 'a')
    file.write('Distances between tracked location and direct step location in pixel:\n')
    file.write('half template size: %.1f, half source size: %.1f\n' % (half_template_size, half_source_size))
    file.write('Mean: %.4f\n' % distances.mean())
    file.write('Min: %.4f\n' % distances.min())
    file.write('Max: %.4f\n' % distances.max())
    file.write('Standard Deviation: %.4f\n' % distances.std())


    np.save(save_path + 'tracked_pts.npy', xy_list)
    np.save(save_path + 'neighbors.npy', neighbors)
    np.save(save_path + 'center_pt.npy', center_pt)



    return distances.mean(), distances.std()

if __name__ == "__main__":
    half_template_size = 30
    half_source_size = 38

    ## Validation simulation
    # half_template_size = 20
    # half_source_size = 68
    mean, std = gaussian_blur_bm(half_template_size=half_template_size, half_source_size=half_source_size, make_plots=True)
