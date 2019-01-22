from moviepy.editor import *
import sys
import pdb
import glob
import os

file_path = 'images/3hierarchy_ccorr_normed/displacement<=20/'
files = glob.glob(file_path+"*.jpg")
files = sorted(files)


clips = [ImageClip(m).set_duration(0.2) for m in files]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("images/3hierarchy_ccorr_normed/displacement<=20/3hierarchy_ccorr_normed.mp4", fps=24)