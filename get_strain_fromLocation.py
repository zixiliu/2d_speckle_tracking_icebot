import numpy as np
import pdb
import cv2 as cv
import glob
import matplotlib.pyplot as plt


manual_file = 'outputs/manuel_pts.npy'
auto_file = 'outputs/auto_pts.npy'

manual_pts = np.load(manual_file)
auto_pts = np.load(auto_file)[1::,:,:]

ori_path = 'images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

def calculate_strain(pts, is_auto = True):
    # End Diastole 
    ED = pts[8,:,:]
    ED_center = ED.mean(axis=0)
    xc, yc = ED_center[0], ED_center[1]

    r0 = []
    c0 = []

    for i, pt in enumerate(ED):
        prev_x, prev_y = ED[i-1][0], ED[i-1][1]
        next_x, next_y = pt[0], pt[1]
        x,y = np.array([(prev_x - next_x)/2.,(prev_y - next_y)/2.])
        arrow = np.array([prev_x-next_x , prev_y-next_y])
        from_center = (x-xc, y-yc)
        from_center = from_center / np.linalg.norm(from_center)

        r = np.abs(np.dot(arrow, from_center))
        r0.append(r)
        c = np.sqrt(np.linalg.norm(arrow)**2 - r**2)
        c0.append(c)

    nframes = pts.shape[0]
    radial = np.zeros((nframes, 6))
    circumferential = np.zeros((nframes, 6))

    for idx, frames in enumerate(pts):
        # img = cv.imread(ori_files[idx+1])
        rs = []
        cs = []
        for i, pt in enumerate(frames):
            prev_x, prev_y = frames[i-1][0], frames[i-1][1]
            next_x, next_y = pt[0], pt[1]
            x,y = np.array([(prev_x - next_x)/2.,(prev_y - next_y)/2.])
            arrow = np.array([prev_x-next_x , prev_y-next_y])
            from_center = (x-xc, y-yc)
            from_center = from_center / np.linalg.norm(from_center)

            r = np.abs(np.dot(arrow, from_center))
            rs.append((r - r0[i])/r0[i])
            c = np.sqrt(np.linalg.norm(arrow)**2 - r**2)
            cs.append((c - c0[i])/c0[i])
            
        rs = np.array(rs)
        cs = np.array(cs)
        # pdb.set_trace()

        if is_auto: 
            radial[idx, 0]=(rs[0]+rs[-3::].sum())/4
            radial[idx, 1]=rs[1:5].mean()
            radial[idx, 2]=rs[5:8].mean()
            radial[idx, 3]=rs[8:12].mean()
            radial[idx, 4]=rs[12:15].mean()
            radial[idx, 5]=rs[15:17].mean()
            print("frame ", idx, 'radial')
            print(radial[idx,:])
            circumferential[idx, 0]=(cs[0]+cs[-3::].sum())/4
            circumferential[idx, 1]=cs[1:5].mean()
            circumferential[idx, 2]=cs[5:8].mean()
            circumferential[idx, 3]=cs[8:12].mean()
            circumferential[idx, 4]=cs[12:15].mean()
            circumferential[idx, 5]=cs[15:17].mean()
            print("frame ", idx, 'circumferential')
            print(circumferential[idx,:])
        else: 
            radial[idx, 0]=(rs[-1]+rs[0:3].sum())/4
            radial[idx, 1]=rs[3:6].mean()
            radial[idx, 2]=rs[6:9].mean()
            radial[idx, 3]=rs[9:13].mean()
            radial[idx, 4]=rs[13:15].mean()
            radial[idx, 5]=rs[15:19].mean()
            print("frame ", idx, 'radial')
            print(radial[idx,:])
            circumferential[idx, 0]=(cs[-1]+cs[0:3].sum())/4
            circumferential[idx, 1]=cs[3:6].mean()
            circumferential[idx, 2]=cs[6:9].mean()
            circumferential[idx, 3]=cs[9:13].mean()
            circumferential[idx, 4]=cs[13:15].mean()
            circumferential[idx, 5]=cs[15:19].mean()
            print("frame ", idx, 'circumferential')
            print(circumferential[idx,:])
    
    # print('r:\n',radial)
    # print('c:\n',circumferential)
    return radial, circumferential

auto_r, auto_c = calculate_strain(auto_pts)
manual_r, manual_c = calculate_strain(manual_pts)

error_r = auto_r.flatten() - manual_r.flatten()
error_c = auto_c.flatten() - manual_c.flatten()

mse_r = ((auto_r.flatten() - manual_r.flatten())**2).mean(axis=0)
mse_c = ((auto_c.flatten() - manual_c.flatten())**2).mean(axis=0)
print('--------------------------------------------------------------------------------')
print('Overall')
print('--------------------------------------------------------------------------------')
print('Mean square error radial strain = '+str(mse_r))
print('Standard deviation error radial strain = '+str(np.std(error_r)))
print('Mean square error circumferential strain = '+str(mse_c))
print('Standard deviation error circumferential strain = '+str(np.std(error_c)))
print('--------------------------------------------------------------------------------')
print('By Region')
print('--------------------------------------------------------------------------------')
print('auto radial max')
print(auto_r.max(axis=0))
print('manual radial max')
print(manual_r.max(axis=0))

print('auto radial min')
print(auto_r.min(axis=0))
print('manual radial min')
print(manual_r.min(axis=0))
print('--------------------------------------------------------------------------------')
print('auto circumferential max')
print(auto_c.max(axis=0))
print('manual circumferential max')
print(manual_c.max(axis=0))

print('auto circumferential min')
print(auto_c.min(axis=0))
print('manual circumferential min')
print(manual_c.min(axis=0))
print('--------------------------------------------------------------------------------')
pdb.set_trace()


# ori_path = 'original/'
# ori_files = glob.glob(ori_path+"*.jpg")
# ori_files = sorted(ori_files)
# file = ori_files[1]
# img = cv.imread(file)
# points = manual_pts[0,:,:]
# for i, pt in enumerate(points): 
#     prev_x, prev_y = int(points[i-1][0]), int(points[i-1][1])
#     x, y = int(pt[0]), int(pt[1])
#     cv.line(img, (prev_x, prev_y), (x, y), (0, 255, 0))
# plt.figure()
# plt.imshow(img)
# plt.show()
