import face_recognition
import cv2
import dlib
import numpy
import time
import os
from matplotlib import pyplot as plt

 ####################################################################################
##  A.face_setup : Attendance using Face Recognition Project
##      1. Detecting face
##      2. Cropping image                           
##      3. Saving it in directory                           
 ####################################################################################


cam = cv2.VideoCapture(0)               # Setting up camera

# Loading cascade files of frontalface, eyes, smile
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


font = cv2.FONT_HERSHEY_SIMPLEX             # setting font
nmbr=0                          # intialising number of images to 0
name = input("enter name: ")                # getting input name
newpath = r'cmpr_img/'+name                 # intialising location of new directory

if not os.path.exists(newpath):             # make new path if path doesn't exist
    os.makedirs(newpath)
# loop runs if capturing has been initialized. 
for i in range(500): 

        ret, img = cam.read()               # starting live video
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting img to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 1) # detecting frontal face

        for (x,y,w,h) in faces: 
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
                roi_gray = gray[y:y+h, x:x+w] 
                roi_color = img[y:y+h, x:x+w] 

                # Detects eyes and smile of different sizes in the input image 
                eyes = eye_cascade.detectMultiScale(roi_gray)
                smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.9, minNeighbors=4, minSize=(10, 10),flags = cv2.CASCADE_SCALE_IMAGE)
                
                #To draw a rectangle in eyes
                if len(eyes) != 0:
                    crop_img=img[y:y+h, x:x+w]      # cropping image to the size of face
                    try:
                        # the 128-dimension face encoding for each face
                        encoding = face_recognition.face_encodings(crop_img)[0]     
                    except (RuntimeError, TypeError, NameError, IndexError):
                        print('Incorrect format')
                        break
                    # saving cropped image to the directory
                    cv2.imwrite('cmpr_img\\'+name+'\\'+str(nmbr)+'img.jpg',crop_img)
                    nmbr+=1             # incrementing the number of images in the dir
                    print(nmbr)
                        
        if nmbr>4:                  # break before number of images in the dir exceeds 5
            break
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff          # wait for Esc key to stop 
        if k == 27: 
            break

cam.release() 

cv2.destroyAllWindows() 
