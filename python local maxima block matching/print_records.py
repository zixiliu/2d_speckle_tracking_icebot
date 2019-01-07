import cv2 as cv
import glob
import pdb
import numpy as np
import matplotlib.pyplot as plt



colors255 = [  (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), 
            (255, 255, 255), (127, 0, 0), (0, 127, 0), (0, 0, 127), (127, 127, 0),
            (0, 127, 127)]

### Read logged points from the manual entry log. Each row is a new trace and the ith (x,y) is to frame i
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1, len(l.split('), '))

entries, frames = file_len('manual_records.txt')
points = np.zeros((entries, frames,2))
file  = open('manual_records.txt', 'r')
for i, line in enumerate(file):
    for j, pt in enumerate(line.split('), ')): 
        x,y = int(pt[1:4]), int(pt[6:9])
        points[i][j] = [x,y]


ori_path = '../original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

record = []
for i in range(1,20):
    new_img = cv.imread(ori_files[i])
    print(ori_files[i])
    for j in range(entries):
        x, y = int(points[j,i-1,0]), int(points[j,i-1,1])
        cv.circle(new_img, (x, y), 2, colors255[j%len(colors255)])
    
    fig = plt.figure(figsize=(64,48),frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(new_img)
    plt.savefig('manual_results/'+ori_files[i][-8::])    