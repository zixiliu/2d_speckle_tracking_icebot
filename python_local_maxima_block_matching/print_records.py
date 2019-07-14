import cv2 as cv
import glob
import pdb
import numpy as np
import matplotlib.pyplot as plt

### Read logged points from the manual entry log. Each row is a new trace and the ith (x,y) is to frame i
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1, len(l.split('), '))

def main():

    record_file_name = 'manual_records3.txt'

    colors255 = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
                (255, 255, 255), (127, 0, 0), (0, 127, 0), (0, 0, 127), (127, 127, 0),
                (0, 127, 127)]

    factor = 480/900.0
    entries, frames = file_len(record_file_name)
    print('num entries: ', entries)
    points = np.zeros((frames, entries,2))
    file  = open(record_file_name, 'r')
    for i, line in enumerate(file):
        for j, pt in enumerate(line.split('), ')):
            x,y = int(pt[1:4]), int(pt[6:9])
            # points[i][j] = [x,y]
            points[j][i] = [x*factor,y*factor]

    np.save('../manuel_pts.npy', points)

    ori_path = '../images/original/'
    # ori_path = 'speckles only/'
    ori_files = glob.glob(ori_path+"*.jpg")
    ori_files = sorted(ori_files)

    # for i in range(1,20):
    for i in range(20):
        new_img = cv.imread(ori_files[i])
    #     pdb.set_trace()
        print(ori_files[i])
        for j in range(entries):
            x, y = int(points[i-1, j, 0]), int(points[i-1, j, 1])
            color = colors255[j%len(colors255)]
            cv.circle(new_img, (x, y), 2, color)
            if j == 0:
                cv.circle(new_img, (x, y), 4, (255, 0, 0))
            if i>=2:
                cv.arrowedLine(new_img, (int(points[i-2,j,0]), int(points[i-2,j,1])), (x, y), color, 2, tipLength=0.3)

            prev_x, prev_y = int(points[i-1][j-1][0]), int(points[i-1][j-1][1])
            cv.line(new_img, (prev_x, prev_y), (x, y), (255,255,0), 1)

        fig = plt.figure(figsize=(64,48),frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(new_img)
        plt.savefig('manual_results2/'+ori_files[i][-8::])


if __name__ == "__main__":
    # execute only if run as a script
    main()