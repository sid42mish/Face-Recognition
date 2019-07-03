import face_recognition
import cv2
import dlib
import numpy
import time
import os
from matplotlib import pyplot as plt

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
nmbr=0

name = input("enter name: ")

newpath = r'cmpr_img/'+name 

if not os.path.exists(newpath):
    os.makedirs(newpath)
# loop runs if capturing has been initialized. 
for i in range(50): 

        ret, img = cam.read() 

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        faces = face_cascade.detectMultiScale(gray, 1.3, 1)
        pic_id = ""

        for (x,y,w,h) in faces: 
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
                roi_gray = gray[y:y+h, x:x+w] 
                roi_color = img[y:y+h, x:x+w] 

                # Detects eyes of different sizes in the input image 
                eyes = eye_cascade.detectMultiScale(roi_gray)
                smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.9, minNeighbors=4, minSize=(10, 10),flags = cv2.CASCADE_SCALE_IMAGE)
                
                #To draw a rectangle in eyes
                if len(eyes) != 0:
                        
                        crop_img=img[y:y+h, x:x+w]
                        cv2.imwrite('cmpr_img\\'+name+'\\'+str(nmbr)+'img.jpg',crop_img)
                        nmbr+=1
                        print(nmbr)
                        #cv2.imshow('img', )
                        #plt.show()
                        '''unknown_face_encoding = face_recognition.face_encodings(crop_img)[0]
                        results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
                        if results[0] == True:
                            print("It's a picture of me!")
                            pic_id="SRK"
                        else:
                            print("It's not a picture of me!")
                            pic_id="not SRK" '''

                
                cv2.putText(img,pic_id,(x,y), font, 1,(255,255,255),2,cv2.LINE_AA)
        # Display an image in a window 
        if nmbr==5:
            break
        cv2.imshow('img',img)
        #time.sleep(4)

        # Wait for Esc key to stop 
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
                break

cam.release() 

cv2.destroyAllWindows() 
