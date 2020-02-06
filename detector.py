import cv2
import numpy as np 
import sqlite3
import os
conn = sqlite3.connect('database.db')
conn2 = sqlite3.connect('database2.db')
c = conn.cursor()
c2 = conn2.cursor()
fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
  print("Please train the data first")
  exit(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)
while True:
  ret, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.1, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
    c.execute("select name from users where id = (?);", (ids,))
    
    result = c.fetchall()
    #conn.commit()
    name = result[0][0]
    if conf < 50:
      cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
      c2.execute('INSERT INTO people (name) VALUES (?)', (name))
      #cmdd="UPDATE name INTO users(name)Values "+str(name)
      #c.execute(cmdd)
      #cmd="INSERT INTO users(id,name) Values("+str(id)+"," +str(name)+")"
      #c.execute(cmd)
    else:
      cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
      c2.execute('INSERT INTO people (name ) values ("No match")')

  cv2.imshow('Face Recognizer',img)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break

cap.release()
conn.commit()
cv2.destroyAllWindows()