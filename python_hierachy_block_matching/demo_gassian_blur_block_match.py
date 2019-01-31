import cv2 as cv
from block_matching import block_match
from trace_trajectory import find_match
import pdb
import glob
import matplotlib.pyplot as plt
import numpy as np

## Global variables 
ori_path = '../images/original/'
ori_files = glob.glob(ori_path+"*.jpg")
ori_files = sorted(ori_files)

def find_match_gaussian_blur(prev_file, next_file, x, y, half_template_size, half_source_size, method, img, first_downsize_by=1, plotit=True):

	ori_file = ori_path + next_file
	ori_prev = ori_path + prev_file

	prev_img = cv.imread(ori_prev)
	prev_gray = cv.cvtColor(prev_img,cv.COLOR_BGR2GRAY)
	next_img = cv.imread(ori_file)
	next_gray = cv.cvtColor(next_img,cv.COLOR_BGR2GRAY)

	tab_size = 7

	prev_gaussian = cv.GaussianBlur(prev_gray,(tab_size,tab_size),0)
	next_gaussian = cv.GaussianBlur(next_gray,(tab_size,tab_size),0)

	## define template
	template = prev_gaussian[y-half_template_size:y+half_template_size+1, x-half_template_size:x+half_template_size+1]

	## define source around the previous match
	yy_ul, yy_lr = y-half_source_size, y+half_source_size+1
	xx_ul, xx_lr =  x-half_source_size, x+half_source_size+1
    
	if yy_ul < 0: 
		yy_ul = 0
	if xx_ul < 0: 
		xx_ul = 0
	if yy_lr > prev_gray.shape[0]-1:
		yy_lr = prev_gray.shape[0]-1
	if xx_lr > prev_gray.shape[1]-1: 
		xx_lr = prev_gray.shape[1]-1

	source = next_gaussian[yy_ul:yy_lr, xx_ul:xx_lr]
	(next_match_x, next_match_y) = block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, x,y)


	if plotit:
		# prev_img_cp = prev_img.copy()
		# next_img_cp = next_img.copy()

		# cv.rectangle(prev_img_cp, (x-half_template_size, y - half_template_size),(x+half_template_size, y+half_template_size), (0,255,0),1)
		cv.rectangle(img, (x-half_template_size, y - half_template_size),(x+half_template_size, y+half_template_size), (0,255,0),1)
		# cv.rectangle(prev_gaussian, (x-half_template_size, y - half_template_size),(x+half_template_size, y+half_template_size), (0,255,0),1)
		# cv.rectangle(next_gaussian, (x-half_template_size, y - half_template_size),(x+half_template_size, y+half_template_size), (0,255,0),1)

		# cv.rectangle(prev_img_cp, (x-half_source_size, y - half_source_size),(x+half_source_size, y+half_source_size), (255,0,0),1)
		cv.rectangle(img, (x-half_source_size, y - half_source_size),(x+half_source_size, y+half_source_size), (255,0,0),1)
		# cv.rectangle(prev_gaussian, (x-half_source_size, y - half_source_size),(x+half_source_size, y+half_source_size), (255,0,0),1)
		# cv.rectangle(next_gaussian, (x-half_source_size, y - half_source_size),(x+half_source_size, y+half_source_size), (255,0,0),1)

		# cv.rectangle(prev_img_cp, (next_match_x-half_template_size, next_match_y - half_template_size),(next_match_x+half_template_size, next_match_y+half_template_size), (0,0,255),1)
		cv.rectangle(img, (next_match_x-half_template_size, next_match_y - half_template_size),(next_match_x+half_template_size, next_match_y+half_template_size), (0,0,255),1)
		# cv.rectangle(prev_gaussian, (next_match_x-half_template_size, next_match_y - half_template_size),(next_match_x+half_template_size, next_match_y+half_template_size), (0,0,255),1)
		# cv.rectangle(next_gaussian, (next_match_x-half_template_size, next_match_y - half_template_size),(next_match_x+half_template_size, next_match_y+half_template_size), (0,0,255),1)

		cv.circle(img, (next_match_x, next_match_y), 3, (255, 0, 0))

		# plt.figure()
		# plt.subplot(221)
		# plt.imshow(prev_img_cp)
		# plt.subplot(222)
		# plt.imshow(next_img_cp)
		# plt.subplot(223)
		# plt.imshow(prev_gaussian)
		# plt.subplot(224)
		# plt.imshow(next_gaussian)
		# plt.show()
		# plt.imsave('../images/gaussian_blur/step0_gaussian_'+prev_file,prev_gaussian)
		# plt.imsave('../images/gaussian_blur/step0_gaussian_'+next_file,next_gaussian)
		# plt.imsave('../images/gaussian_blur/step0_'+prev_file,prev_img_cp)
		# plt.imsave('../images/gaussian_blur/step0_'+next_file,next_img_cp)

	# pdb.set_trace()

	# ## define template
	# next_match_x, next_match_y = int(next_match_x), int(next_match_y)
	# yy_ul, yy_lr = next_match_y-half_template_size, next_match_y+half_template_size+1
	# xx_ul, xx_lr =  next_match_x-half_template_size, next_match_x+half_template_size+1

	# if yy_ul < 0: 
	# 	yy_ul = 0
	# if xx_ul < 0: 
	# 	xx_ul = 0
	# if yy_lr > prev_gray.shape[0]-1:
	# 	yy_lr = prev_gray.shape[0]-1
	# if xx_lr > prev_gray.shape[1]-1: 
	# 	xx_lr = prev_gray.shape[1]-1
	
	# half_source_size = half_template_size
	# half_template_size = 4
	# template = prev_gray[next_match_y-half_template_size : next_match_y+half_template_size+1, \
	# 					 next_match_x-half_template_size : next_match_x+half_template_size+1]

	# source = next_gray[yy_ul:yy_lr, xx_ul:xx_lr]
	# # pdb.set_trace()
	# (next_match_x, next_match_y) = block_match(source, template, method, half_template_size, yy_ul, yy_lr, xx_ul, xx_lr, next_match_x,next_match_y)

	

	# if plotit:
	# 	prev_img_cp = prev_img.copy()
	# 	next_img_cp = next_img.copy()
	# 	try:
	# 		x, y = int(next_match_x), int(next_match_y)
	# 	except:
	# 		pdb.set_trace()


	# 	cv.rectangle(prev_img_cp, (x-half_template_size, y - half_template_size),(x+half_template_size, y+half_template_size), (0,255,0),1)
	# 	cv.rectangle(next_img_cp, (x-half_template_size, y - half_template_size),(x+half_template_size, y+half_template_size), (0,255,0),1)
		

	# 	cv.rectangle(prev_img_cp, (x-half_source_size, y - half_source_size),(x+half_source_size, y+half_source_size), (255,0,0),1)
	# 	cv.rectangle(next_img_cp, (x-half_source_size, y - half_source_size),(x+half_source_size, y+half_source_size), (255,0,0),1)
		

	# 	cv.rectangle(prev_img_cp, (next_match_x-half_template_size, next_match_y - half_template_size),(next_match_x+half_template_size, next_match_y+half_template_size), (0,0,255),1)
	# 	cv.rectangle(next_img_cp, (next_match_x-half_template_size, next_match_y - half_template_size),(next_match_x+half_template_size, next_match_y+half_template_size), (0,0,255),1)

	# 	print('save plot step 1')
	# 	plt.imsave('../images/gaussian_blur/step1_'+prev_file,prev_img_cp)
	# 	plt.imsave('../images/gaussian_blur/step1_'+next_file,next_img_cp)

	# pdb.set_trace()
	return next_match_x, next_match_y, img


f1 = '4067.jpg'
f2 = '4068.jpg'

match_dictionary = {}
method = cv.TM_CCORR_NORMED
half_template_size = 18
# x, y = 100*4, 45*4 #50,70. 55,70
# x, y = (584*640/1200, 240*640/1200)
x, y = (439, 353)
x, y = x*640/1200, y*640/1200
xys = [[584, 240], [565, 266], [534, 285], [487, 321], [439, 353], [402, 389], [412, 448], \
		[458, 484], [490, 510], [548, 556], [569, 576], [600, 622], [635, 585], [668, 490], \
		[686, 424], [711, 369], [751, 342], [756, 318], [716, 271], [634, 288]]
num_pts = len(xys)
for i, xy in enumerate(xys): 
	x, y = xy[0], xy[1]
	x, y = x*640/1200, y*640/1200
	xys[i] = [x, y]

first_downsize_by = 1
# half_source_size = half_template_size * 2
half_source_size = 25

xy_list = np.zeros((len(xys), 28, 2))

xy_list[:,0,:] = xys


for ii, i in enumerate(range(4051,4078)): #4078
	f1 = str(i)+'.jpg'
	f2 = str(i+1)+'.jpg'
	print('----')
	print(f2)

	next_img = cv.imread(ori_path+f2)
	# pdb.set_trace()
	for j in range(num_pts):
		x, y = int(xy_list[j, ii, 0]), int(xy_list[j, ii, 1])
		
		next_match_x, next_match_y, next_img = find_match_gaussian_blur(f1, f2, x, y, half_template_size, half_source_size, method, next_img, plotit= False)
		x, y = next_match_x/first_downsize_by, next_match_y/first_downsize_by
		
		xy_list[j,ii+1,:] =[x, y] 

		for temp in range(ii): 	
			prev_x, prev_y = xy_list[j, temp, 0], xy_list[j, temp,1] 
			new_x, new_y = xy_list[j, temp+1, 0], xy_list[j, temp+1,1] 
			cv.line(next_img, (int(prev_x*first_downsize_by), int(prev_y*first_downsize_by)), \
				    (int(new_x*first_downsize_by), int(new_y*first_downsize_by)), (0,0,255), 1)
			cv.circle(next_img, (int(new_x*first_downsize_by), int(new_y*first_downsize_by)), 3, (255, 255, 0))
		cv.circle(next_img, (int(next_match_x*first_downsize_by), int(next_match_y*first_downsize_by)), 3, (255, 0, 0))
	# print('red: '+str((x*first_downsize_by, y*first_downsize_by)))

	for pp in range(num_pts): 
		x, y = int(xys[pp][0]), int(xys[pp][1])
		cv.circle(next_img, (x, y), 3, (0, 255, 0))

	# fig = plt.figure(figsize=(14,8),frameon=False)
	# ax = plt.Axes(fig, [0., 0., 1., 1.])
	# ax.set_axis_off()
	# fig.add_axes(ax)
	# ax.imshow(ori_img)
	# plt.show()
	# plt.savefig('../images/3hierarchy_ccorr_normed/1pt_trace/'+f2)
	plt.imsave('../images/gaussian_blur/'+f2, next_img)

pdb.set_trace()