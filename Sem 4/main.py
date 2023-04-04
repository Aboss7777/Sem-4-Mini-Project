
import cv2
import numpy as np
import face_recognition

imgDetect = face_recognition.load_image_file('image/Ayush.jpg')
imgDetect = cv2.cvtColor(imgDetect, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('image/Shakunt.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgDetect)[0]
encodeDetect = face_recognition.face_encodings(imgDetect)[0]
cv2.rectangle(imgDetect, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (155, 0, 255), 2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (155, 0, 255), 2)

results = face_recognition.compare_faces([encodeDetect], encodeTest)
faceDis = face_recognition.face_distance([encodeDetect], encodeTest)
print(results, faceDis)

cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (100,0,255), 2)

cv2.imshow('Ayush', imgDetect)
cv2.imshow('Shakunt', imgTest)
cv2.waitKey(0)