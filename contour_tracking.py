import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import sys
import pdb
import glob
import colorsys
from skimage.feature import peak_local_max

### Create N distinct colors for tracking
# HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
# RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
# colors = []
# for i in RGB_tuples: 
#     colors.append((int(i[0]*255), int(i[1]*255), int(i[2]*255)))

colors = [  (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
            (255, 255, 255), (127, 0, 0), (0, 127, 0), (0, 0, 127), (127, 127, 0),
            (0, 127, 127)]
N = len(colors)

# pdb.set_trace()

num_colors = 0

class MyContour: 
    def __init__(self, area, centroid, color_index, contour):
        self.area = area
        self.centroid = centroid
        self.x = centroid[0]
        self.y = centroid[1]
        self.color_index = color_index
        self.contour = contour



def get_contours(imgray):
    ### Threshold it so it becomes binary
    imgray = cv.cvtColor(imgray,cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(imgray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    ### Find contours and get a list of my_contour objects 
    _, contours, _ = cv.findContours(thresh,cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse = True)
    
    return contours

def get_my_contour(imgray): 
    contours = get_contours(imgray)
    my_contours = []
    for i, c in enumerate(contours): 
        c_moments = cv.moments(c)
        if c_moments['m00'] != 0:
            this_contour = MyContour(c_moments['m00'], [c_moments['m10']/c_moments['m00'],c_moments['m01']/c_moments['m00']], i, c)
            my_contours.append(this_contour)
    return my_contours

def helper_get_distance(x1, y1, x2, y2):
	'''Get distance between two keypoints'''
	return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def draw_contour_by_dilation(dilation_filename, ori_filename, prev_contours): 
    global num_colors, N

    ### Read the image
    src = cv.imread(dilation_filename)
    ori = cv.imread(ori_filename)


    ## track local peaks in the original file
    ori_gray = cv.cvtColor(ori,cv.COLOR_BGR2GRAY)
    # coordinates = peak_local_max(ori_gray, min_distance=1)
 
    imgray = src.copy()    
    ### find 
    if prev_contours == None:
        my_contours = get_my_contour(imgray)
        # num_colors += len(my_contours)
        num_colors += 1
    else: 
        ### compare each contour to the previous contours and find a match if exists
        contours = get_contours(imgray)
        my_contours = []
        # for c in contours:
        c = contours[1]
        is_new_contour = False
        c_moments = cv.moments(c)
        if c_moments['m00'] != 0:
            area = c_moments['m00']
            (x, y) = (c_moments['m10']/c_moments['m00'],c_moments['m01']/c_moments['m00'])

            ### Find potential matches of centroids within 20 pixels of the current contour
            potential_match = []
            for prev_c in prev_contours: 
                if helper_get_distance(x, y, prev_c.x, prev_c.y) <= 40: 
                    potential_match.append(prev_c)
            ### Find a match that has the most similar area
            if len(potential_match) > 0: 
                closest = min(potential_match, key=lambda x:abs(x.area-area))                

                # if abs(closest.area - area) < 0.4 * area:
                if ((0.4 * area < closest.area) & (closest.area < (4 * area))):
                    this_contour = MyContour(area, [x, y], closest.color_index, c)
                    my_contours.append(this_contour)
                    # pdb.set_trace()
                    cv.arrowedLine(ori, (int(this_contour.x), int(this_contour.y)), 
                                            (int(closest.x), int(closest.y)), (255, 255, 0), 3, tipLength=0.3)

                else: 
                    is_new_contour = True
            else: 
                is_new_contour = True

            if is_new_contour:
                this_contour = MyContour(area, [x, y], num_colors, c)
                num_colors += 1
                my_contours.append(this_contour)

    
    # Draw circles on the local peaks 
    # TWO layer hierachy
    coordinates_large = peak_local_max(ori_gray, footprint = np.ones((10, 10)))
    # coordinates_large = peak_local_max(ori_gray, min_distance = 10)

    # delaunay triangulation
    size = ori.shape
    rect = (0, 0, size[1], size[0])
    subdiv  = cv.Subdiv2D(rect)

    for item in coordinates_large: 
        y, x  = item[0], item[1]
        pt_in_contour = cv.pointPolygonTest(my_contours[0].contour, (x, y), False) 
        if pt_in_contour == 1: 
            # cv.circle(ori, (x, y), 2, (255, 0, 0))
            subdiv.insert((x, y))
    
    triangleList = subdiv.getTriangleList()
    size = ori.shape
    r = (0, 0, size[1], size[0])
    delaunay_color = (255, 255, 0)
    
    for t in triangleList :
        # pdb.set_trace()
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            # cv.line(ori, pt1, pt2, delaunay_color, 1)
            # cv.line(ori, pt2, pt3, delaunay_color, 1)
            # cv.line(ori, pt3, pt1, delaunay_color, 1)
            pass
    
    for item in coordinates_large: 
        y, x  = item[0], item[1]
        # pt_in_contour = cv.pointPolygonTest(my_contours[0].contour, (x, y), False) 
        # if pt_in_contour == 1: 
        cv.circle(ori, (x, y), 2, (255, 0, 0))
            # ori[y, x] = np.array([255,0,0])

    # coordinates_small = peak_local_max(ori_gray, footprint = np.ones((3, 3)))
    # for item in coordinates_small: 
    #     if item in coordinates_large: 
    #         pass
    #     else:
    #         y, x= item[0], item[1]
    #         pt_in_contour = cv.pointPolygonTest(my_contours[0].contour, (x, y), False) 
    #         if pt_in_contour == 1: 
    #             cv.circle(ori, (x, y), 2, (255, 255, 0))
    

    plt.figure(figsize=(50,40))
    for i, c in enumerate(my_contours):        
        color = colors[c.color_index % N]
        cv.drawContours(ori, [c.contour], 0, color, 1)
    plt.imshow(ori)
    # plt.show()
    # plt.imsave('contour/'+dilation_filename[-8:-1], ori)
    plt.savefig("first_contour/"+dilation_filename[-8::])

    return my_contours



dilation_path = 'dilation/'
ori_path = 'original/'

dilation_files = glob.glob(dilation_path+"*.jpg")
dilation_files = sorted(dilation_files)

ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

# The very first frame
i = 0
contours = draw_contour_by_dilation(dilation_files[i], ori_files[i], None)
# The second frame forward
for i, file in enumerate(dilation_files):
    print(file)
    contours = draw_contour_by_dilation(dilation_files[i], ori_files[i], contours)