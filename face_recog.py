import face_recognition
import cv2
import dlib
import numpy as np
import os

 ####################################################################################
##	A.face_recog : Attendance using Face Recognition Project
##		1. Detecting face
##		2. Classifying faces by comparing unknown face with known faces						
##		3. Marking attendance for checked classes
 ####################################################################################

cam = cv2.VideoCapture(0)       # setting up camera

names = os.listdir('cmpr_img')  # getting classes from directory
classes = []

nmbr_of_img = 5                 # number of images in a class
min_imgs = 2                    # minimum hits required

for i in names:
    encodings = []
    for j in range (nmbr_of_img):
        # loading images from dir
        img = face_recognition.load_image_file("cmpr_img/"+i+"/"+str(j)+"img.jpg")
        # 128-dimension face encoding for each face
        img_encoding = face_recognition.face_encodings(img)[0]
        encodings.append(img_encoding)  # appending encoded images in list 'encodings'
    classes.append(encodings)           # appending encodings in 2-D list 'classes'
    
attendance =  {key: 0 for key in names} # initialising attendance list
print(attendance)
detector = dlib.get_frontal_face_detector() # detecting frontal face
color_green = (0,255,0)                     # color of box
line_width = 3                              # width of box

# during live video feed
while True:
    ret_val, img = cam.read()               # getting frame from live feed
    dets = detector(img)                    # detecting frontal faces in frame

    for det in dets:     
        # Crop Face from the whole image frame
        crop_img = img[det.left():det.right(), det.top():det.bottom()]
        
        # Create boundary box as(x, y, width, height) instead of four corner points
        bbox = (det.left()-5, det.top()-5, det.right() - det.left()+10 ,det.bottom() - det.top()+10)

        # Detecting face in frame
        unknown_picture = face_recognition.face_locations(img)
        flag = 0                            # initialising flag
        count = [0]*len(names)              # initialising count
        try:
            unknown_face_encoding = face_recognition.face_encodings(img,unknown_picture)[0]
        
            for i in range (nmbr_of_img):
                # Draw rectangle around Face with color:Green and line width: 3pt
                cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
                for j in range (len(names)):
                    # comparing unknown faces with known classes
                    results1 = face_recognition.compare_faces([classes[j][i]], unknown_face_encoding, tolerance = 0.45)
                    if results1[0] == True:         # checking matches
                    	count[j] += 1               # increment count of that class
    
                    if count[j] > min_imgs:         # checking if count of that class exceeds min images
                    	break

                # Project image in video frame
                if results1[0] == True and count[j] > min_imgs: # checking if matched and min images condition
                    flag = 1                        # setting flag to 1
                    attendance[names[j]] = 1        # marking present to that class
                    # putting name of that class beside box
                    cv2.putText(img, names[j], (det.left(), det.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    break

            if not flag:
                # setting that image to 'none' class
                cv2.putText(img, "none", (det.left(), det.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        except (RuntimeError, TypeError, NameError, IndexError):
            print("no image found")

        
    cv2.imshow('LiveStream', img)                   # displaying live stream on screen
 
    
   
    # Exit if ESC pressed    
    k = cv2.waitKey(1) & 0xff
    if k == 97:
        print(attendance)                           # printing attendance if 'a' is pressed
    if k == 27 : 
    	print(attendance)                           # printing final attendance when ESC is pressed
    	break
