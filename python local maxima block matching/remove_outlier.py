import cv2 as cv
import pdb
import matplotlib.pyplot as plt
import numpy as np



################################################################################
########## Helper Functions
################################################################################
# Check if a point is inside a rectangle
def helper_rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def helper_get_delaunay_neighbors(edge_list): 
    neighbor_dict = {}
    for entry in edge_list: 
        x1, y1, x2, y2 = entry[0], entry[1], entry[2], entry[3]
        if (x1, y1) in neighbor_dict: 
            neighbor_dict[(x1, y1)].append((x2, y2))
        else: 
            neighbor_dict[(x1, y1)] = [(x2, y2)]
    return neighbor_dict

def helper_get_distance(x1, y1, x2, y2):
	## Get distance between two keypoints
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def helper_get_residual(x,y, v_x, v_y, neighbors, velocity_dict): 
    if len(neighbors)> 1: 
        dxy = []
        vel_x = []
        vel_y = []
        for pt in neighbors: 
            try:
                neighbor_x, neighbor_y = pt[0], pt[1]
        
                if pt in velocity_dict:
                    dxy.append(helper_get_distance(x, y, neighbor_x, neighbor_y))
                    vel_x.append(velocity_dict[pt][0])
                    vel_y.append(velocity_dict[pt][1])
                else: 
                    print("Problem! neighbor pt not found in the frame!")
                    return 0
                    # pdb.set_trace()
                    
            except: 
                print(pt)
                pdb.set_trace()
        median_d = np.median(np.array(dxy))    
        epi = (-median_d+ np.sqrt(median_d**2+0.4))/2

        r_x = abs(v_x - median_d) / ( np.median(np.abs(vel_x - median_d)) + epi)
        r_y = abs(v_y - median_d) / ( np.median(np.abs(vel_y - median_d)) + epi)
        r = np.sqrt(r_x**2 + r_y**2)
    else: 
        return 0

    return r

def helper_get_normalized_residual(x,y, v_x, v_y, neighbors, velocity_dict): 
    if len(neighbors)> 1: 
        dxy = []
        # dx = []
        # dy = []
        vel_x = []
        vel_y = []
        for pt in neighbors: 
            try:
                neighbor_x, neighbor_y = pt[0], pt[1]
        
                if pt in velocity_dict:
                    dxy.append(helper_get_distance(x, y, neighbor_x, neighbor_y))
                    # dx.append(abs(x - neighbor_x))
                    # dy.append(abs(y - neighbor_y))
                    vel_x.append(velocity_dict[pt][0])
                    vel_y.append(velocity_dict[pt][1])
                else: 
                    print("Problem! neighbor pt not found in the frame!")
                    return 0
                    # pdb.set_trace()
                    
            except: 
                print(pt)
                pdb.set_trace()
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

def helper_is_outier(r, method='delaunay'): 
    # print(r)
    if (method=='delaunay') & (r > 3.5):
        return True
    if (method=='neighbor_pixel') & (r > 1.5): 
        return True
    else: 
        return False

def helper_get_pixel_neighbors(velocity_dict, neighbor_halfsize):
    sorted_keys = sorted(velocity_dict.keys())
    neighbors = {}
    
    for i, key in enumerate(sorted_keys):
        
        neighbors[key] = []
        ## Going back
        j = i-1
        while j > 0: 
            prev_xy = sorted_keys[j]
            # if helper_get_distance(key[0], key[1], prev_xy[0], prev_xy[1]) <= neighbor_halfsize: 
            if abs(key[0] - prev_xy[0]) <= neighbor_halfsize: 
                if abs(key[1] - prev_xy[1]) <= neighbor_halfsize: 
                    neighbors[key].append(prev_xy)
                j -= 1
            else: 
                break
        ## Going forward
        j = i+1
        while j < len(sorted_keys): 
            prev_xy = sorted_keys[j]
            # if helper_get_distance(key[0], key[1], prev_xy[0], prev_xy[1]) <= neighbor_halfsize: 
            if abs(key[0] - prev_xy[0]) <= neighbor_halfsize: 
                if abs(key[1] - prev_xy[1]) <= neighbor_halfsize: 
                    neighbors[key].append(prev_xy)
                j+=1
            else: 
                break
        if len(neighbors[key]) == 0:
            del neighbors[key]
        # else:
        #     # print(len(neighbors[key]))
        #     pdb.set_trace()
    return neighbors

def helper_remove_outlier(match_dict, neighbor_halfsize):
    
    velocity_dict = {}
    for match in match_dict:
        prev_x, prev_y = match[0], match[1]
        new_xy = match_dict[match]
        new_x, new_y = new_xy[0], new_xy[1]
        velocity_dict[new_xy] = ((new_x - prev_x)/40.0, (new_y - prev_y)/40.0)
    
    neighbor_dict = helper_get_pixel_neighbors(velocity_dict, neighbor_halfsize)
    # print("pts with neighbors: ", len(neighbor_dict.keys()))

    pts_removed = 0
    for i, pt in enumerate(sorted(neighbor_dict.keys())):
        try: 
            vxy = velocity_dict[pt]
            x, y, v_x, v_y = pt[0], pt[1], vxy[0], vxy[1]
            neighbors = neighbor_dict[pt]
            
            if len(neighbors) > 1:
                r = helper_get_normalized_residual(x,y, v_x, v_y, neighbors, velocity_dict)
                # print(r)
                # r = helper_get_residual(x,y, v_x, v_y, neighbors, velocity_dict)
                if helper_is_outier(r, method='neighbor_pixel'):
                    # ## draw a square around the point removed
                    # cv.rectangle(new_img, (x-neighbor_halfsize, y-neighbor_halfsize), 
                    #                     (x+neighbor_halfsize, y+neighbor_halfsize), (255, 0, 0))
                    # for n in neighbors: 
                    #     n_x, n_y = n[0], n[1]
                    #     sqs = 2
                    #     cv.rectangle(new_img, (n_x-sqs, n_y-sqs), (n_x+sqs, n_y+sqs), (0, 255, 0))

                    # if return_pts_removed:
                    # for key, value in match_dict.items():
                    #     if value == pt:
                    #         cv.arrowedLine(new_img, (key[0], key[1]), (value[0], value[1]), (255, 0, 0), 1, tipLength=0.3)
                            # points_removed[key] = value
                            
                    match_dict = {key: value for key, value in match_dict.items() if value != pt}
                    pts_removed += 1
                # else: 
                    
                #     # if (i > 111) & (i < 113):
                #     # if (i > 510) & (i < 513):
                #     if i == 511:
                #         # print('======')
                #         print(i, r, x, y, velocity_dict[pt]) 
                #         cv.rectangle(new_img, (x-neighbor_halfsize, y-neighbor_halfsize), 
                #                             (x+neighbor_halfsize, y+neighbor_halfsize), (255, 0, 0))
                #         for n in neighbors: 
                #             n_x, n_y = n[0], n[1]
                #             # sqs = 2
                #             print(n, velocity_dict[n])
                #             # cv.rectangle(new_img, (n_x-sqs, n_y-sqs), (n_x+sqs, n_y+sqs), (0, 255, 0))
                #             # cv.circle(new_img, (n_x, n_y), 2, (0, 255, 0))
            else:
                match_dict = {key: value for key, value in match_dict.items() if value != pt}
        except: 
            pass
    # print('After removing the outliers: ', len(match_dict))
    # print('points removed: ', pts_removed)

    return pts_removed, match_dict

################################################################################
########## Private Functions
################################################################################

def priv_remove_by_delaunay(match_dict, img_size, new_img, return_pts_removed=False): 
    if return_pts_removed:
        points_removed = {}

    print('Before removing the outliers: ', len(match_dict))

    # delaunay triangulation
    rect = (0, 0, img_size[1], img_size[0])
    subdiv  = cv.Subdiv2D(rect)
    
    velocity_dict = {}
    for match in match_dict:
        prev_x, prev_y = match[0], match[1]
        new_xy = match_dict[match]
        new_x, new_y = new_xy[0], new_xy[1]
        subdiv.insert(new_xy)
        velocity_dict[new_xy] = ((new_x - prev_x)/40.0, (new_y - prev_y)/40.0)

    triangleList = subdiv.getTriangleList()
    edge_list = subdiv.getEdgeList()
    neighbor_dict = helper_get_delaunay_neighbors(edge_list)

    for pt in neighbor_dict.keys():
        try: 
            vxy = velocity_dict[pt]
            x, y, v_x, v_y = pt[0], pt[1], vxy[0], vxy[1]
            neighbors = neighbor_dict[pt]
            r = helper_get_normalized_residual(x,y, v_x, v_y, neighbors, velocity_dict)
            # r = helper_get_residual(x,y, v_x, v_y, neighbors, velocity_dict)
            
            if helper_is_outier(r, method='delaunay'):
                
                if return_pts_removed:
                        for key, value in match_dict.items():
                            if value == pt:
                                points_removed[key] = value
                match_dict = {key: value for key, value in match_dict.items() if value != pt}
        except: 
            pass
    print('After removing the outliers: ', len(match_dict))

    # delaunay_color = (255, 255, 0)
    # for t in triangleList :
    #     pt1 = (int(t[0]), int(t[1]))
    #     pt2 = (t[2], t[3])
    #     pt3 = (t[4], t[5])

        # if helper_rect_contains(rect, pt1) and helper_rect_contains(rect, pt2) and helper_rect_contains(rect, pt3) :
        #     cv.line(new_img, pt1, pt2, delaunay_color, 1)
        #     cv.line(new_img, pt2, pt3, delaunay_color, 1)
        #     cv.line(new_img, pt3, pt1, delaunay_color, 1)    
    return match_dict, new_img

def priv_remove_by_neighborPixel(match_dict, img_size, new_img, neighbor_halfsize = 25, return_pts_removed=False):
    print('Before removing the outliers: ', len(match_dict))    
    if return_pts_removed:
        points_removed = {}

    pts_removed = np.Inf
    num_rounds = 0
    while pts_removed > 0: 
        print(len(match_dict))
        pts_removed, match_dict = helper_remove_outlier(match_dict, neighbor_halfsize)
        num_rounds += 1
    
    # pts_removed, match_dict = helper_remove_outlier(match_dict, neighbor_halfsize)


    print("num rounds of removal: ", num_rounds)

    # fig = plt.figure(figsize=(10,10),frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(new_img)
    # plt.show()
    # pdb.set_trace()

    if return_pts_removed: 
        return points_removed, new_img
    else: 
        return match_dict, new_img

################################################################################
########## Public Functions
################################################################################

def remove_outlier(match_dict, img_size, new_img, method='delaunay'): 
    if method == 'delaunay': 
        return priv_remove_by_delaunay(match_dict, img_size, new_img)
    if method == 'neighbor_pixel': 
        return priv_remove_by_neighborPixel(match_dict, img_size, new_img, return_pts_removed=False)