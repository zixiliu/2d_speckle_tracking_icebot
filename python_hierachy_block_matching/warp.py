import cv2
import numpy as np
import pdb 
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import scipy.interpolate as interpolate



# img = np.uint8(np.random.rand(8, 8)*255)
# print(img)
# print('---------------')
# map_y = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
# map_x = np.array([[5, 6], [7, 10], [3.6, 5]], dtype=np.float32)
# mapped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
# print(mapped_img)

# pdb.set_trace()

f = '../images/original/4051.jpg'
img = cv2.imread(f)

boundary = [[320-17+3, 48], [320-292+3, 330], [320+3, 450], [320+292+3, 330], [320+17+3, 48]]
# for i, xy in enumerate(boundary): 
#     x, y = xy[0] , xy[1] 
#     boundary[i] = [x, y]
#     cv2.circle(img, (x, y), 3, (0,255,0))
print(boundary)
# plt.figure()
# plt.imshow(img)
# plt.show()

x = np.array([31,  323, 615])
y = np.array([330, 450, 330])
t, c, k = interpolate.splrep(x, y, s=0, k=2)
N = 200#50
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)


topx = np.linspace(306,340, N)
M = 300#50

warp_x = np.zeros((M, N), dtype=np.float32)
warp_y = np.zeros((M, N), dtype=np.float32)

for i, x in enumerate(xx): 
#     cv2.circle(img, (int(x), int(spline(x))), 3, (0,255,0))
#     cv2.circle(img, (int(topx[i]), 48), 3, (0,255,0))
#     cv2.line(img, (int(topx[i]), 48), (int(x), int(spline(x))), (0,255,0),1)
    interp_x = np.linspace(int(topx[i]), int(x), M)
    interp_y = np.linspace(48, int(spline(x)), M)
    for j in range(M): 
        # cv2.circle(img, (int(interp_x[j]), int(interp_y[j])), 3, (255,0,0))
        warp_x[j, i] = interp_x[j]
        warp_y[j, i] = interp_y[j]
            
# plt.figure()
# plt.imshow(img)
# plt.show()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.figure()
plt.subplot(121)
plt.imshow(gray, cmap='gray')

mapped_img = cv2.remap(gray, warp_x, warp_y, cv2.INTER_LINEAR)
# print(mapped_img)

plt.subplot(122)
plt.imshow(mapped_img, cmap='gray')
plt.show()

np.save('warpx.npy', warp_x)
np.save('warpy.npy', warp_y)

pdb.set_trace()
