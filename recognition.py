
import numpy as np
import face_recognition
import os
from datetime import datetime
import cv2
import dlib

path = '/Users/kaivanvisaria/PycharmProjects/rdbmsia2/img'
images = []
classnames = []
myList = os.listdir(path)
print(myList)
#path for images
for cls in myList:
   curimg = cv2.imread(f'{path}/{cls}')
   images.append(curimg)
   classnames.append(os.path.splitext(cls)[0])
print(classnames)
def Findencodings(images):
    encodelist = []
    #loop through all images
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
def markattendance(name):
    with open('/Users/kaivanvisaria/PycharmProjects/rdbmsia2/attendance.csv','r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%m/%d/%Y, %H:%M:%S:')
            f.writelines(f'\n{name},{dtstring}')

        print(mydatalist)
markattendance('a')
encodelistknown = Findencodings(images)
print(len(encodelistknown))
cap = cv2.VideoCapture(0)
while True:
   sucess, img = cap.read()
   imgs = cv2.resize(img,(0,0),None,0.25,0.25)
   imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

   facecurfr= face_recognition.face_locations(imgs)
   encodecurfr = face_recognition.face_encodings(imgs,facecurfr)
   #loop through in current frame
   for encodeface,faceloc in zip(encodecurfr,facecurfr):
       matches = face_recognition.compare_faces(encodelistknown,encodeface)
       facedis = face_recognition.face_distance(encodelistknown,encodeface)
       print(facedis)
       #minimum distance
       matchindex = np.argmin(facedis)

       #if true
       if matches[matchindex]:
           name = classnames[matchindex].upper()
           print(name)
           y1,x2,y2,x1 = faceloc
           y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
           cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
           cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
           cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
           markattendance(name)

   cv2.imshow('webcam', img)
   cv2.waitKey(1)













