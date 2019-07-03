import face_recognition
import cv2
import dlib
import numpy as np
import os

cam = cv2.VideoCapture(0)

names = os.listdir('cmpr_img')
classes = []
for i in names:
    encodings = []
    for j in range (5):
        img = face_recognition.load_image_file("cmpr_img/"+i+"/"+str(j)+"img.jpg")
        img_encoding = face_recognition.face_encodings(img)[0]
        encodings.append(img_encoding)
    classes.append(encodings)

detector = dlib.get_frontal_face_detector()
color_green = (0,255,0)
line_width = 3


while True:
    ret_val, img = cam.read()
    #rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #tracker = cv2.TrackerBoosting_create()
    dets = detector(img)
    for det in dets:
        
        # Draw rectangle around Face with color:Green and line width: 3pt
        
        
        # Crop Face from the whole image frame
        crop_img = img[det.left():det.right(), det.top():det.bottom()]
        
        # Create boundary box as(x, y, width, height) instead of four corner points
        bbox = (det.left()-5, det.top()-5, det.right() - det.left()+10 ,det.bottom() - det.top()+10)

        unknown_picture = face_recognition.face_locations(img)
        flag = 0
        try:
            unknown_face_encoding = face_recognition.face_encodings(img,unknown_picture)[0]
        
            for i in range (5):
                cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
                for j in range (len(names)):
                    
                    results1 = face_recognition.compare_faces([classes[j][i]], unknown_face_encoding, tolerance = 0.50)
                    if results1[0] == True:
                        #print(results1)
                        break

                        # Project image in video frame
                if results1[0] == True:
                    flag = 1                        
                    cv2.putText(img, names[j], (det.left(), det.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

                    break
            if not flag:
                cv2.putText(img, "none", (det.left(), det.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        except (RuntimeError, TypeError, NameError, IndexError):
            print("no image found")

        
    cv2.imshow('LiveStream', img)
        
    # Add tracker for face boundaries in image
 
    
   
    # Exit if ESC pressed    
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
