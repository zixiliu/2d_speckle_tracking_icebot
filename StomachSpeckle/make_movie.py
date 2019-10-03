from moviepy.editor import *
import sys
import pdb
import glob
import os


def jpg_to_mp4(file_path, video_name):
    files = glob.glob(file_path+'*.jpg')
    files = sorted(files)

    clips = [ImageClip(m).set_duration(0.05) for m in files] #0.2
    # pdb.set_trace()
    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(video_name, fps=30)#fps=24)


if __name__ == "__main__":
    file_path = '/Users/zixiliu/Documents/Daichi/20190926_Images/20190621104753_S8_probe/tracking/'
    video_file_name = '/Users/zixiliu/Documents/Daichi/20190926_Images/20190621104753_S8_probe/tracking/tracked.mp4'

    jpg_to_mp4(file_path, video_file_name)



