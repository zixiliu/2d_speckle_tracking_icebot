import numpy as np 
import cv2 as cv 
import glob
import matplotlib.pyplot as plt
import pdb
# from localMax_blockMatching import get_local_max
from skimage.feature import peak_local_max

ori_path = '../images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)
# file = ori_files[1]




def get_local_max(img, footprintsize=5): 
    imgray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    coordinates = peak_local_max(imgray, threshold_abs = 100, footprint = np.ones((footprintsize, footprintsize)))
    peaks = []
    for xy in coordinates: 
        x, y = xy[0], xy[1]
        peaks.append(imgray[x, y])
    sort_index = np.argsort(peaks)
    sort_index = sort_index[::-1]
    
    coordinates = coordinates[sort_index]
    return coordinates

def mark_peaks(img, footprintsize=5, marksize=1):
    coordinates = get_local_max(img, footprintsize)
    for xy in coordinates: 
        x, y = int(xy[0]), int(xy[1])
        if marksize > 1:
            cv.circle(img, (y, x), marksize, (255, 0, 0))
        else:
            img[x, y] = np.array([255, 0, 0])
    return img

def plot_downsampled_imgs(file):

    img = cv.imread(file)
    img2 = cv.pyrDown(img)
    img4 = cv.pyrDown(img2)
    img8 = cv.pyrDown(img4)
    img16 = cv.pyrDown(img8)
    img32 = cv.pyrDown(img16)

    plt.imsave('../images/downsamplePlot/down_by_2/'+file[-8::], img2)
    plt.imsave('../images/downsamplePlot/down_by_4/'+file[-8::], img4)
    plt.imsave('../images/downsamplePlot/down_by_8/'+file[-8::], img8)
    plt.imsave('../images/downsamplePlot/down_by_16/'+file[-8::], img16)

    img = mark_peaks(img, marksize=4)
    img2 = mark_peaks(img2, marksize=2)
    img4 = mark_peaks(img4, footprintsize=5)
    img8 = mark_peaks(img8, footprintsize=3)
    # img16 = mark_peaks(img16, footprintsize=3)
    # img32 = mark_peaks(img32)

    plt.figure()
    plt.imshow(img2)
    plt.savefig('../images/downsamplePlot/down_by_2/w_peak/'+file[-8::])

    plt.figure()
    plt.imshow(img4)
    plt.savefig('../images/downsamplePlot/down_by_4/w_peak/'+file[-8::])

    plt.figure()
    plt.imshow(img8)
    plt.savefig('../images/downsamplePlot/down_by_8/w_peak/'+file[-8::])

    # def plot_im(ax, img, title): 
    #     ax.imshow(img)
    #     ax.set_title(title, fontsize=12)
    # fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(12,8))
    # img_list = [img, img2, img4, img8]
    # title_list = ['original', 'down by 2', 'down by 4', 'down by 8']
    # for i, ax in enumerate(axs.flatten()):
    #     plot_im(ax, img_list[i], title_list[i])



    # plt.figure(constrained_layout=True)    
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.title('original')
    # plt.subplot(222)
    # plt.imshow(img2)
    # plt.title('down by 2')

    # plt.subplot(223)
    # plt.imshow(img4)
    # plt.title('down by 4')
    # plt.subplot(224)
    # plt.imshow(img8)
    # plt.title('down by 8')

    # plt.show()
    print('../images/downsamplePlot/'+file[-8::])
    # plt.savefig('../images/downsamplePlot/'+file[-8::], bbox_inches='tight', pad_inches=0, transparent=True)

for file in ori_files:
    print(file)
    plot_downsampled_imgs(file)


# file1 = '../images/downsamplePlot/down_by_8/savefig_4051.jpg'
# file2 = '../images/downsamplePlot/down_by_8/4051.jpg'

# img1 = cv.imread(file1)
# img2 = cv.imread(file2)

# plt.subplot(121)
# plt.imshow(img1)
# plt.subplot(122)
# plt.imshow(img2)
# plt.show()

# pdb.set_trace()