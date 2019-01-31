from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image, ImageTk
import cv2 as cv
import glob
import pdb


green = '#000fff000'
red = '#fff000000'

def get_coordinate(File, prev_file, prev_x, prev_y):
    
    global x, y
    global green, red
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
    img = ImageTk.PhotoImage(Image.open(File))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    # Circle around previous_xy
    canvas.create_rectangle(prev_x-2, prev_y-2, prev_x+2, prev_y+2, outline=green)

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

    ori_path = '../images/original/'
    # ori_path = 'speckles only/'
    ori_files = glob.glob(ori_path+"*.jpg")
    ori_files = sorted(ori_files)

    record = []
    x, y = 0,0
    i = 0
    x,y = get_coordinate(ori_files[i], None, x, y)
    for i in range(1, 6):
        print(ori_files[i])
        x,y = get_coordinate(ori_files[i], ori_files[i-1], x, y)
        record.append((x, y))

    print(record)
    return record

def record_one_speckle():
    file  = open('manual_records3.txt', 'a')
    record = get_recording()
    file.write(str(record)[1:-1]+'\n')
    # pdb.set_trace()

record_one_speckle()