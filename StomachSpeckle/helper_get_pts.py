from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import cv2 as cv
import glob
import pdb
import numpy as np

def helper_get_coordinate(File):
    global pts_to_track
    green = '#000fff000'
    red = '#fff000000'
    img = cv.imread(File)
    h, w = img.shape[:2]

    x, y = 0, 0
    root = Tk()
    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set, width=w, height=h)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    im_temp = Image.open(File)
    img = ImageTk.PhotoImage(im_temp)
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    # Circle around previous_xy
    if pts_to_track != []:
        for item in pts_to_track:
            xx,yy = item[0], item[1]
            canvas.create_rectangle(xx-2, yy-2, xx+2, yy+2, outline=red)

    #function to be called when mouse is clicked
    def printcoords(event):
        global x, y
        x, y = event.x, event.y
        #outputting x and y coords to console
        print (x,y)
        pts_to_track.append([x, y])
        root.destroy()

    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()
    return (x, y)


def helper_record_speckles_in_one_frame(img_file):
    '''To record multiple points in a single frame (for instance for initial points to track)'''
    pts_to_track = []
    global pts_to_track

    # Get 20 points
    for i in range(20): 
        x,y = helper_get_coordinate(img_file)

    return pts_to_track

def manual_select_and_save(img_file, save_file):
    pts_to_track = helper_record_speckles_in_one_frame(img_file)
    np.save(save_file, pts_to_track)


# if __name__ == "__main__":
    # file = '/Users/zixiliu/my_git_repos/my_howe_lab/Echos/Zeo file/IM_1637_copy_jpg/1.jpg'
    # pts_to_track = helper_record_speckles_in_one_frame(file)
    # print(pts_to_track)
