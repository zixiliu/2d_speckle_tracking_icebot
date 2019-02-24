## This is an implementation based on https://www.learnopencv.com/histogram-of-oriented-gradients/
import cv2 as cv
import numpy as np
import glob
import pdb


def get_hist(gray):

    if len(gray.shape) != 2: 
        print("Expecting a single channel gray image")
        raise ValueError
    gray = np.float32(gray) / 255.0
    
    # Calculate gradient 
    try:
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=1)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=1)
    except:
        pdb.set_trace()

    # Python Calculate gradient magnitude and direction ( in degrees ) 
    mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

    (row, col) = mag.shape
    hist = np.zeros(9)

    for r in range(row): 
        for c in range(col):
            direction = angle[r,c] % 180 ## non-polarized
            if direction <= 20: 
                to_left = (direction/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[0] += to_left
                hist[1] += to_right
            elif direction <= 40: 
                to_left = ((direction-20)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[1] += to_left
                hist[2] += to_right
            elif direction <= 60: 
                to_left = ((direction-40)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[2] += to_left
                hist[3] += to_right
            elif direction <= 80: 
                to_left = ((direction-60)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[3] += to_left
                hist[4] += to_right
            elif direction <= 100: 
                to_left = ((direction-80)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[4] += to_left
                hist[5] += to_right
            elif direction <= 120: 
                to_left = ((direction-100)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[5] += to_left
                hist[6] += to_right
            elif direction <= 140: 
                to_left = ((direction-120)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[6] += to_left
                hist[7] += to_right
            elif direction <= 160: 
                to_left = ((direction-140)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[7] += to_left
                hist[8] += to_right
            elif direction <= 180: 
                to_left = ((direction-160)/20.) * mag[r, c]
                to_right = mag[r, c] - to_left
                hist[8] += to_left
                hist[0] += to_right
            else:
                raise ValueError

    return hist

def hog_match(template, source):
    trow, tcol = template.shape
    srow, scol = source.shape

    template_hist= get_hist(template)

    source_hist_matrix = np.zeros((srow-trow+1, scol-tcol+1))

    for yshift in range(srow-trow):
        for xshift in range(scol-tcol):
            block = source[yshift:yshift+trow, xshift:xshift+tcol]
            # pdb.set_trace()
            block_hist = get_hist(block)
            pearson_coeff = np.corrcoef(template_hist, block_hist)[0,1]
            if pearson_coeff == np.nan:
                pearson_coeff = 1
            source_hist_matrix[yshift, xshift] = pearson_coeff
    source_hist_matrix = np.nan_to_num(source_hist_matrix)
    xy = np.where(source_hist_matrix==source_hist_matrix.max())
    # pdb.set_trace()
    try:
        x, y = xy[0][0], xy[1][0]
    except:
        pdb.set_trace()
    # pdb.set_trace()
    return [[y, x]]




# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    path = '../images/downsamplePlot/down_by_8/'
    files = glob.glob(path+"*.jpg")
    files = sorted(files)
    f1 = '4067.jpg'
    file1 = path+f1
    img = cv.imread(file1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # hist = get_hist(gray)
    # print(hist)


    target = gray[26-7:26+7, 26-7:26+7]
    source =  gray[26-10:26+10, 26-10:26+10]
    hog_match(target, source)