from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import cv2 as cv
import glob
import pdb
import numpy as np



green = '#000fff000'
red = '#fff000000'

record_file_name = 'manual_records.txt'
record = []

def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    if i > 0: 
        return i + 1, len(l.split('), '))   
    else: 
        return -1, -1

entries, frames = file_len(record_file_name)
print('entries = %.2f, frames=%.2f'%(entries, frames))
has_previous_log = False
if entries > 0:
    has_previous_log = True
    points = np.zeros((entries, frames,2))
    file  = open(record_file_name, 'r')
    for i, line in enumerate(file):
        for j, pt in enumerate(line.split('), ')): 
            x,y = int(pt[1:4]), int(pt[6:9])
            points[i][j] = [x,y]
else: 
    points = []

def get_coordinate(File, prev_file, prev_x, prev_y, frame_num):
    
    global x, y
    global green, red
    global record, points
    global has_previous_log

    x, y = 0, 0
    if prev_file != None: 
        prev_img = cv.imread(prev_file)
        cv.circle(prev_img, (prev_x, prev_y), 4, (0, 255, 0))
        cv.imshow('dst_rt', prev_img)
    
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set, width=1200, height=900)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    # File = askopenfilename(parent=root, initialdir="../original",title='Choose an image.')
    
    im_temp = Image.open(File)
    im_temp = im_temp.resize((1200, 900), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(im_temp)
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    # Circle around previous_xy
    canvas.create_rectangle(prev_x-3, prev_y-3, prev_x+3, prev_y+3, outline=green)
    for item in record: 
        xx,yy = item[0], item[1]
        canvas.create_rectangle(xx-2, yy-2, xx+2, yy+2, outline=red)
    if has_previous_log:
        for item in points[:, frame_num]:
            xx,yy = item[0], item[1]
            canvas.create_rectangle(xx-2, yy-2, xx+2, yy+2, outline=green)


    #function to be called when mouse is clicked
    def printcoords(event):
        global x, y
        x, y = event.x, event.y
        #outputting x and y coords to console
        print (x,y)
        
        root.destroy()

    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()    
    if prev_file != None: 
        cv.destroyAllWindows()
    return (x, y)

def get_recording():
    # File = "../original/4051.jpg"
    # x,y = get_coordinate(File)
    # display_prev(File, x, y)
    # pdb.set_trace()

    global record 
    ori_path = '../images/original/'
    # ori_path = 'speckles only/'
    ori_files = glob.glob(ori_path+"*.jpg")
    ori_files = sorted(ori_files)
    
    x, y = 0,0
    i = 0
    x,y = get_coordinate(ori_files[i], None, x, y, 0)
    for i in range(0, 20):
        print(ori_files[i])
        if i == 0:
            x,y = get_coordinate(ori_files[i], None, x, y, i)
        else:
            x,y = get_coordinate(ori_files[i], ori_files[i-1], x, y, i)
        record.append((x, y))

    print(record)
    return record

def record_one_speckle():
    file  = open(record_file_name, 'a')
    record = get_recording()
    file.write(str(record)[1:-1]+'\n')
    


record_one_speckle()