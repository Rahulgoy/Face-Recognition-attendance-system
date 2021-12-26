import cv2
from face_recognition.api import face_encodings
import numpy
import face_recognition
import os
import numpy as np

img1 = cv2.imread("Images/Rahul Goyal.jpeg")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.imread("Rahul.jpeg")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

face_loc1 = face_recognition.face_locations(img1)[0]
face_enc1 = face_recognition.face_encodings(img1)

face_loc2 = face_recognition.face_locations(img2)[0]
face_enc2 = face_recognition.face_encodings(img2)

result = face_recognition.compare_faces(face_enc1,face_enc2[0])
print(result)