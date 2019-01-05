import cv2 as cv
import pdb
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

def helper_get_neighbors(edge_list): 
    neighbor_dict = {}
    for entry in edge_list: 
        x1, y1, x2, y2 = entry[0], entry[1], entry[2], entry[3]
        if (x1, y1) in neighbor_dict: 
            neighbor_dict[(x1, y1)].append((x2, y2))
        else: 
            neighbor_dict[(x1, y1)] = [(x2, y2)]
    return neighbor_dict

def helper_get_median(neighbors): 
    return None

def helper_get_distance(x1, y1, x2, y2):
	## Get distance between two keypoints
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)

################################################################################
########## Public Functions
################################################################################
def get_normalized_residual(x,y, v_x, v_y, neighbors, velocity_dict): 
    if len(neighbors)> 1: 
        dx = []
        dy = []
        vel_x = []
        vel_y = []
        for pt in neighbors: 
            try:
                neighbor_x, neighbor_y = pt[0], pt[1]
        
                if pt in velocity_dict:
                    dx.append(abs(x - neighbor_x))
                    dy.append(abs(y - neighbor_y))
                    vel_x.append(velocity_dict[pt][0])
                    vel_y.append(velocity_dict[pt][1])
                else: 
                    print("Problem! neighbor pt not found in the frame!")
                    return 0
                    # pdb.set_trace()
                    
            except: 
                print(pt)
                pdb.set_trace()
        median_d_x = np.median(np.array(dx))    
        median_d_y = np.median(np.array(dy))    

        epi_x = (-median_d_x+ np.sqrt(median_d_x**2+0.4))/2
        epi_y = (-median_d_y+ np.sqrt(median_d_y**2+0.4))/2

        norm_median_x = np.median(vel_x/(dx+epi_x))
        norm_median_y = np.median(vel_y/(dy+epi_y))

        r_x = abs(v_x/(median_d_x + epi_x) - norm_median_x) / ( np.median(v_x/(dx + epi_x) - norm_median_x) + epi_x)
        r_y = abs(v_y/(median_d_y + epi_y) - norm_median_y) / ( np.median(v_y/(dy + epi_y) - norm_median_y) + epi_y)
        r = np.sqrt(r_x**2 + r_y**2)
    else: 
        return 100

    return r

def is_outier(r): 
    if r > 3.5:
        return True
    else: 
        return False

def remove_outlier(match_dict, img_size, new_img): 
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
        velocity_dict[new_xy] = ((new_x - prev_x)/40, (new_y - prev_y)/40)

    triangleList = subdiv.getTriangleList()
    edge_list = subdiv.getEdgeList()
    neighbor_dict = helper_get_neighbors(edge_list)

    for pt in neighbor_dict.keys():
        try: 
            vxy = velocity_dict[pt]
            x, y, v_x, v_y = pt[0], pt[1], vxy[0], vxy[1]
            neighbors = neighbor_dict[pt]
            r = get_normalized_residual(x,y, v_x, v_y, neighbors, velocity_dict)
            if is_outier(r):
                match_dict = {key: value for key, value in match_dict.items() if value != pt}
        except: 
            pass
    print('After removing the outliers: ', len(match_dict))
    pdb.set_trace() 


    delaunay_color = (255, 255, 0)
    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if helper_rect_contains(rect, pt1) and helper_rect_contains(rect, pt2) and helper_rect_contains(rect, pt3) :
            cv.line(new_img, pt1, pt2, delaunay_color, 1)
            cv.line(new_img, pt2, pt3, delaunay_color, 1)
            cv.line(new_img, pt3, pt1, delaunay_color, 1)    


    return match_dict, new_img