from moviepy.editor import *
import sys
import pdb
import glob
import os

# file_path = 'images/original/'
file_path = 'images/gaussian_blur/no warp no gaussian/'
files = glob.glob(file_path+'*.jpg')
files = sorted(files)


clips = [ImageClip(m).set_duration(0.05) for m in files] #0.2
# pdb.set_trace()
concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("images/gaussian_blur/no warp no gaussian/no_warp_gaussian.mp4", fps=24)#fps=24)