from moviepy.editor import *
import sys
import pdb
import glob
import os

file_path = 'laplacian_blob/'
files = glob.glob(file_path+"*.jpg")
files = sorted(files)


clips = [ImageClip(m).set_duration(0.2) for m in files]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("laplacian_blob.mp4", fps=24)