import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pdb

################################################################################
## Global variables
################################################################################
# pts = np.load('images/block_matching/tracked_pts.npy')
# pts = pts[:,0:22,:] # strip out the 0s in the end
# neighbors = np.load('images/block_matching/neighbors.npy')
# neighbors = neighbors.item() # dictionary of indices of pts within distances of 128 pixels
# center_pt = np.load('images/block_matching/center_pt.npy')
# center_pt = center_pt[0:22,:] # strip out the 0s in the end
ori_path='validation_simulation/'
# ori_path = 'images/original/'
save_img_path = ori_path + 'strain/'
save_path = ori_path

# starting_img_num = 4065 #4051
# ending_img_num = 4084 #4072
starting_img_num = 0
ending_img_num = 19

pts = np.load(ori_path+'tracked_pts.npy')
pts = pts[:,0:(ending_img_num-starting_img_num),:] # strip out the 0s in the end
neighbors = np.load(ori_path+'neighbors.npy')
neighbors = neighbors.item() # dictionary of indices of pts within distances of 128 pixels
center_pt = np.load(ori_path+'center_pt.npy')
center_pt = center_pt[0:(ending_img_num-starting_img_num),:] # strip out the 0s in the end

with_warp = False
if with_warp:
    warp_x = np.load('python_hierachy_block_matching/warpx.npy')
    warp_y = np.load('python_hierachy_block_matching/warpy.npy')




border = 20
delaunay_color = (255, 255, 0)
big_triangle_color = (255, 0, 0)
################################################################################
## Helper function
################################################################################
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

'''https://www.programiz.com/python-programming/examples/area-triangle'''
def get_area(pt1, pt2, pt3):
    a = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    b = np.sqrt((pt2[0]-pt3[0])**2 + (pt2[1]-pt3[1])**2)
    c = np.sqrt((pt1[0]-pt3[0])**2 + (pt1[1]-pt3[1])**2)
    s = (a+b+c)/2.
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    return area

def get_distance(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

#Use base cmap to create transparent
mycmap = transparent_cmap(plt.cm.Reds)

################################################################################
## Define triangle
################################################################################
class Triangle:
    def __init__(self, pt1, pt2, pt3, v1, v2, v3):
        self.pt1 = (int(pt1[0]),int(pt1[1]))
        self.pt2 = (int(pt2[0]),int(pt2[1]))
        self.pt3 = (int(pt3[0]),int(pt3[1]))
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    ''' Returns True if successfully replaced and False otherwise'''
    def replace(self, old_pt, new_pt, new_v):
        old_pt = (int(old_pt[0]),int(old_pt[1]))
        new_pt = (int(new_pt[0]),int(new_pt[1]))
        if self.pt1 == old_pt:
            self.pt1 = new_pt
            self.v1 = new_v
            return True

        elif self.pt2 == old_pt:
            self.pt2 = new_pt
            self.v2 = new_v
            return True
        elif self.pt3 == old_pt:
            self.pt3 = new_pt
            self.v3 = new_v
            return True
        print("about to return error in replacing a point in triangle")
        pdb.set_trace()
        return False

    def boundary(self):
        left_most_x = int(min(self.pt1[0], self.pt2[0], self.pt3[0]))
        # left_most_y = min(self.pt1[1], self.pt2[1], self.pt3[1])

        right_most_x = int(max(self.pt1[0], self.pt2[0], self.pt3[0]))
        # right_most_y = min(self.pt1[1], self.pt2[1], self.pt3[1])

        # top_most_x = min(self.pt1[0], self.pt2[0], self.pt3[0])
        top_most_y = int(min(self.pt1[1], self.pt2[1], self.pt3[1]))

        # bottom_most_x = min(self.pt1[0], self.pt2[0], self.pt3[0])
        bottom_most_y = int(max(self.pt1[1], self.pt2[1], self.pt3[1]))

        return ((left_most_x, top_most_y), (right_most_x, bottom_most_y))

    def print_val(self):
        print(self.pt1, self.pt2, self.pt3, self.v1, self.v2, self.v3)

    def is_inside(self, pt):
        tol = 1e-3

        A = get_area(self.pt1, self.pt2, self.pt3)
        A1 = get_area(self.pt1, self.pt2, pt)
        A2 = get_area(self.pt1, self.pt3, pt)
        A3 = get_area(self.pt2, self.pt3, pt)
        if(np.abs(A -(A1 + A2 + A3)) < tol):
            return True
        else:
            # print(self.pt1, self.pt2, self.pt3, '|', pt)
            # pdb.set_trace()
            return False

    def get_v(self, pt):
        '''https://codeplea.com/triangular-interpolation'''
        if self.is_inside(pt):
            d1 = get_distance(self.pt1, pt)
            d2 = get_distance(self.pt2, pt)
            d3 = get_distance(self.pt3, pt)

            if d1 == 0:
                return (True, self.v1)
            if d2 == 0:
                return (True, self.v2)
            if d3 == 0:
                return (True, self.v3)
            w1 = 1./d1
            w2 = 1./d2
            w3 = 1./d3
            wsum = w1+w2+w3
            v = ((self.v1[0]*w1+self.v2[0]*w2+self.v3[0]*w3)/wsum, (self.v1[1]*w1+self.v2[1]*w2+self.v3[1]*w3)/wsum)
            return (True, v)
        else:
            return (False, (np.Inf, np.Inf))

################################################################################
# f = ori_path + '4051.jpg'
f = ori_path + str(starting_img_num) + '.jpg'
ori = cv.imread(f)
if with_warp:
    warp = cv.remap(ori, warp_x, warp_y, cv.INTER_LINEAR)
    warp = cv.copyMakeBorder(warp, border, border, border, border, cv.BORDER_CONSTANT)
else:
    warp = ori.copy()
## Delaunay triangularization of points in the first frame
size = ori.shape
rect = (0, 0, size[1], size[0])
subdiv  = cv.Subdiv2D(rect)
for item in pts[:,0,:]:
    x, y  = item[0], item[1]
    subdiv.insert((x, y))
triangleList = subdiv.getTriangleList() # The first pair of numbers are the x and y position of the first vertex of the triangle.
                                        # The second pair of numbers are for the second vertex.
                                        # The third pair of numbers are for the third vertex of the triangle.

triangleDict = {}

## Define triangles
for t in triangleList :
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3) :
        if get_area(pt1, pt2, pt3) <= 40:
            cv.line(warp, pt1, pt2, delaunay_color, 1)
            cv.line(warp, pt2, pt3, delaunay_color, 1)
            cv.line(warp, pt3, pt1, delaunay_color, 1)
            # define triangle
            this_triangle = Triangle(pt1, pt2, pt3, (0,0), (0,0), (0,0))
            try:
                triangleDict[pt1].append(this_triangle)
            except KeyError:
                triangleDict[pt1] = [this_triangle]
            try:
                triangleDict[pt2].append(this_triangle)
            except KeyError:
                triangleDict[pt2] = [this_triangle]
            try:
                triangleDict[pt3].append(this_triangle)
            except KeyError:
                triangleDict[pt3] = [this_triangle]
        else:
            cv.line(warp, pt1, pt2, big_triangle_color, 1)
            cv.line(warp, pt2, pt3, big_triangle_color, 1)
            cv.line(warp, pt3, pt1, big_triangle_color, 1)

# pdb.set_trace()
# plt.imshow(warp)
# plt.show()
# pdb.set_trace()

## displacement field: frame number, every frame, for each pixel, [dx,dy]
displacement_field = np.zeros((pts.shape[1]-1, warp.shape[0], warp.shape[1], 2)) # for each pixel, [dx,dy]
epsilon_x = np.zeros((pts.shape[1]-1, warp.shape[0], warp.shape[1]))
epsilon_y = np.zeros((pts.shape[1]-1, warp.shape[0], warp.shape[1]))
gamma_xy = np.zeros((pts.shape[1]-1, warp.shape[0], warp.shape[1]))
radial = np.zeros((pts.shape[1]-1, warp.shape[0], warp.shape[1]))
circumferential = np.zeros((pts.shape[1]-1, warp.shape[0], warp.shape[1]))
# pdb.set_trace()
## Get displacement and update key
pdb.set_trace()
for i in range(1, pts.shape[1]): # for each frame
    for j in range(pts.shape[0]):
        point = (pts[j, i, 0], pts[j, i, 1])
        prev_point = (pts[j, i-1, 0], pts[j, i-1, 1])
        vp = (point[0]-prev_point[0], point[1]-prev_point[1])

        try: # not every point belongs to a triangle of accepted area
            triangleDict[point] = triangleDict[prev_point]
            del triangleDict[prev_point]
            for t in triangleDict[point]:
                rtn = t.replace(prev_point, point, vp)
                if rtn == False:
                    print("couldn't replace!")
                    pdb.set_trace()
        except:
            pass


    ## for all the unique triangles in this frame
    all_triangles = triangleDict.values()
    flat_list = [item for sublist in all_triangles for item in sublist]
    myset = set(flat_list)
    all_triangles = list(myset)

    for t in all_triangles:
        # pdb.set_trace()
        ((left_most_x, top_most_y), (right_most_x, bottom_most_y)) = t.boundary()
        for xx in range(left_most_x, right_most_x+1):
            for yy in range(top_most_y, bottom_most_y+1):
                res = t.get_v((xx, yy))
                if res[0] == True:
                    # pdb.set_trace()
                    v = res[1]
                    displacement_field[i-1, yy, xx, :] = [v[0], v[1]]
                    # pdb.set_trace()
        # pdb.set_trace()
    ## smooth out the left out pixels
    # pdb.set_trace()
    # displacement_field[i-1, :, :, 0] = cv.GaussianBlur(displacement_field[i-1, :, :, 0], (11,11),0)
    # displacement_field[i-1, :, :, 1] = cv.GaussianBlur(displacement_field[i-1, :, :, 1], (11,11),0)
    center = center_pt[i,:]


    # pdb.set_trace()
    # plt.subplot(131)
    # plt.imshow(warp)
    # plt.subplot(132)
    # plt.imshow(displacement_field[0,:,:,0])
    # plt.subplot(133)
    # plt.imshow(displacement_field[0,:,:,1])
    # plt.show()

    ## first order approximation
    for xx in range(warp.shape[0]-1):
        for yy in range(warp.shape[1]-1):
            try:
                epsilon_x[i-1, xx, yy] = displacement_field[i-1, xx+1, yy, 0] - displacement_field[i-1, xx, yy, 0]
                epsilon_y[i-1, xx, yy] = displacement_field[i-1, xx, yy+1, 1] - displacement_field[i-1, xx, yy, 1]
                gamma_xy[i-1, xx, yy] = displacement_field[i-1, xx, yy+1, 0] - displacement_field[i-1, xx, yy, 0] + \
                                        displacement_field[i-1, xx+1, yy, 1] - displacement_field[i-1, xx, yy, 1]
                from_center = [xx-center[0], yy-center[1]]
                r = from_center[0]**2 + from_center[1]**2
                if np.sqrt(r) != 0:
                    cos_theta = from_center[0]/r
                    sin_theta = from_center[1]/r#np.sqrt(1-cos_theta**2)#from_center[1]/r
                    radial[i-1, xx, yy] = epsilon_x[i-1, xx, yy] * cos_theta**2 + gamma_xy[i-1, xx, yy]*cos_theta*sin_theta + epsilon_y[i-1, xx, yy]*sin_theta**2
                    circumferential[i-1, xx, yy] = epsilon_x[i-1, xx, yy] * sin_theta**2 - gamma_xy[i-1, xx, yy]*cos_theta*sin_theta + epsilon_y[i-1, xx, yy]*cos_theta**2

                    if np.isnan(radial[i-1, xx, yy]):
                        print("nan")
                        pdb.set_trace()
            except:
                pdb.set_trace()

    f = ori_path + str(starting_img_num+i)+'.jpg'
    print(f)
    img = cv.imread(f)
    if with_warp:
        warp = cv.remap(img, warp_x, warp_y, cv.INTER_LINEAR)
        warp = cv.copyMakeBorder(warp, border, border, border, border, cv.BORDER_CONSTANT)

    ## Make strain masks
    mask_radial = np.zeros((warp.shape[0], warp.shape[1], 3))
    print('r',radial[i-1,:,:].max(), radial[i-1,:,:].min())
    if radial[i-1,:,:].max() == 0:
        pdb.set_trace()
    mask_radial[:,:,0] = np.where(radial[i-1,:,:]>=0,radial[i-1,:,:],0)/0.005*255#/12.5*255
    mask_radial[:,:,2] = np.where(radial[i-1,:,:]<0,-radial[i-1,:,:],0)/0.005*255#/12.5*255

    mask_circ = np.zeros((warp.shape[0], warp.shape[1], 3))
    mask_circ[:,:,0] = np.where(circumferential[i-1,:,:]>=0,circumferential[i-1,:,:],0)/0.005*255#/12.5*255
    mask_circ[:,:,2] = np.where(circumferential[i-1,:,:]<0,-circumferential[i-1,:,:],0)/0.005*255#/12.5*255
    print('c',circumferential[i-1,:,:].max(), circumferential[i-1,:,:].min())

    triangle_mask = np.zeros((warp.shape[0], warp.shape[1], 3))
    ## draw triangles
    line_color = (255,255,0)
    for t in all_triangles:
        pt1,pt2,pt3 = t.pt1, t.pt2, t.pt3
        cv.line(triangle_mask, pt1, pt2, line_color, 1)
        cv.line(triangle_mask, pt3, pt2, line_color, 1)
        cv.line(triangle_mask, pt1, pt3, line_color, 1)
        # cv.line(mask_radial, pt1, pt2, line_color, 1)
        # cv.line(mask_radial, pt3, pt2, line_color, 1)
        # cv.line(mask_radial, pt1, pt3, line_color, 1)
        # cv.line(mask_circ, pt1, pt2, line_color, 1)
        # cv.line(mask_circ, pt3, pt2, line_color, 1)
        # cv.line(mask_circ, pt1, pt3, line_color, 1)

    ## Draw points
    # pt_mask = np.zeros((warp.shape[0], warp.shape[1], 3))
    for pt_idx in range(pts.shape[0]):
        point = pts[pt_idx,i,:]
        prev_pt = pts[pt_idx,i-1,:]
        cv.circle(mask_radial, (int(point[0]), int(point[1])), 2, (255,255,0))
        cv.arrowedLine(mask_radial, (int(prev_pt[0]), int(prev_pt[1])), (int(point[0]), int(point[1])), (0,255,0), 1, tipLength=0.3)
        cv.circle(mask_circ, (int(point[0]), int(point[1])), 2, (255,255,0))
        cv.arrowedLine(mask_circ, (int(prev_pt[0]), int(prev_pt[1])), (int(point[0]), int(point[1])), (0,255,0), 1, tipLength=0.3)

    cv.circle(mask_radial, (int(center[0]), int(center[1])), 3, (0,255,0))
    cv.circle(mask_circ, (int(center[0]), int(center[1])), 3, (0,255,0))


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[25,10])
    # fig, ax1 = plt.subplots(1, 1, figsize=[10,15])
    ax1.imshow(warp)
    # cb = ax1.contourf(radial[i-1,:,:], cmap='bwr',alpha = 0.5 , extend='both', vmin=-12.5, vmax=12.5)
    # # cb = ax1.contourf(dismag, cmap='bwr',alpha = 0.5 , extend='both')
    # plt.colorbar(cb,ax=ax1)
    # ax1.imshow(triangle_mask,alpha=0.4)
    ax1.imshow(mask_radial,alpha=0.6)
    ax1.set_title('Radial Strain',fontsize=16)

    ax2.imshow(warp)
    # ax2.imshow(triangle_mask,alpha=0.4)
    ax2.imshow(mask_circ,alpha=0.6)
    ax2.set_title('Circumferential Strain',fontsize=16)

    ax3.imshow(warp)
    ax3.imshow(triangle_mask,alpha=0.5)
    ax3.set_title('Trianglization',fontsize=16)

    plt.savefig(save_img_path+str(starting_img_num+i)+'.jpg')

    mask_dis0 = np.zeros((warp.shape[0], warp.shape[1], 3))
    mask_dis0[:,:,0] = np.where(displacement_field[i-1,:,:,0]>=0,displacement_field[i-1,:,:,0],0)/8.*255
    mask_dis0[:,:,2] = np.where(displacement_field[i-1,:,:,0]<0,-displacement_field[i-1,:,:,0],0)/8.*255
    # print(mask_dis0.max(), mask_dis0.min())

    mask_dis1 = np.zeros((warp.shape[0], warp.shape[1], 3))
    mask_dis1[:,:,0] = np.where(displacement_field[i-1,:,:,1]>=0,displacement_field[i-1,:,:,1],0)/8.*255
    mask_dis1[:,:,2] = np.where(displacement_field[i-1,:,:,1]<0,-displacement_field[i-1,:,:,1],0)/8.*255
    # print(mask_dis1.max(), mask_dis1.min())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[15,10])
    ax1.imshow(warp)
    ax1.imshow(mask_dis0,alpha=0.5)
    ax1.set_title('displacement field x direction')
    ax2.imshow(warp)
    ax2.imshow(mask_dis1,alpha=0.5)
    ax2.set_title('displacement field y direction')
    plt.savefig(ori_path+'displacement_field/'+str(starting_img_num+i)+'.jpg')
    # pdb.set_trace()



print('Done')
pdb.set_trace()









# if __name__ == "__main__":
#     pass