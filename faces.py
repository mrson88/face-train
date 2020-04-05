import numpy as np
import cv2,os
import matplotlib as plt
import pickle

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer =cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {"person_name",1}
with open("labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        #print(roi_gray)
        #print(roi_color)
        id_,conf=recognizer.predict(roi_gray)
        if conf>=45: #and conf<=85:
            print(labels[id_])
            print(id_)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        img_item='my_image.png'
        cv2.imwrite(img_item,roi_gray)

        color = (255, 0, 0)
        stoke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stoke)


    cv2.imshow('frame',frame)
    if cv2.waitKey(20)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
