import numpy as np
import face_recognition
import os
from datetime import datetime
import cv2
import dlib

#load the image
imgutsavi = face_recognition.load_image_file('/Users/kaivanvisaria/PycharmProjects/rdbmsia2/basicimg/utsavi.JPG')
#convert to grayscale
imgutsavi = cv2.cvtColor(imgutsavi,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('/Users/kaivanvisaria/PycharmProjects/rdbmsia2/basicimg/aayushi.JPG')
imgtest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2RGB)
#detect the face
#single image so first element
faceLoc = face_recognition.face_locations(imgutsavi)[0]
print(faceLoc)

#get the encodings
encodeUtsavi = face_recognition.face_encodings(imgutsavi)[0]
#draw rectangle
cv2.rectangle(imgutsavi,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoctest = face_recognition.face_locations(imgtest)[0]
encodeUtsavitest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,255),2)

#compare and calculate distance(backend linear svm)
results = face_recognition.compare_faces([encodeUtsavi], encodeUtsavitest)

faceDis = face_recognition.face_distance([encodeUtsavi], encodeUtsavitest)
print(results,faceDis)
cv2.putText(imgtest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Utsavi Visaria',imgutsavi)
cv2.imshow('Utsavi Test',imgtest)
cv2.waitKey(0)
