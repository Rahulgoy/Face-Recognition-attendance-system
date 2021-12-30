import cv2
import numpy
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

path = 'Images'

images = []
classNames = []
myList = os.listdir(path)
# print(myList)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])


print(len(images))
def findencodings(images):
    encodeList = []
    count=0
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        # print(encode)
        if encode:
            # print(encode)
            encodeList.append(encode[0])
        else:
            print(classNames[count])
            # print(img)
        count=count+1
    return encodeList


try:
    # face_name= np.loadtxt('face_names.csv',delimiter=',')
    with open('face_names.pkl','rb') as f: face_name = pickle.load(f)
    with open('encodings.pkl','rb') as f: encodeListKnown = pickle.load(f)
    print("Try block")
    if len(face_name)<len(classNames):
        encodeListKnown = findencodings(images)
        classNames=face_name
except:
    print("except block")
    encodeListKnown = findencodings(images)
    with open('face_names.pkl','wb') as f: 
        pickle.dump(classNames, f)
    with open('encodings.pkl','wb') as f: 
        pickle.dump(encodeListKnown, f)



# print(classNames)
# print(face_name)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


# encodeListKnown = findencodings(images)
# data = np.asarray(encodeListKnown)
# print(encodeListKnown)
# np.savetxt('test.csv', data, delimiter=',')

# face_names = np.asarray(classNames)
# np.savetxt('face_names.csv', face_names, delimiter=',')
# encodeListKnown= np.loadtxt('test.csv',delimiter=',')
# print(data_new)


print(encodeListKnown)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    # img = cv2.imread("Rahul.jpeg")
    red_img = cv2.resize(img, (0,0),None,0.25,0.25)
    red_img = cv2.cvtColor(red_img,cv2.COLOR_BGR2RGB)

    face_location = face_recognition.face_locations(red_img)
    # print(face_location)
    encode_of_curr_frame = face_recognition.face_encodings(red_img,face_location)

    for encode_face, face_loc in zip(encode_of_curr_frame,face_location):
        matches = face_recognition.compare_faces(encodeListKnown,encode_face)
        face_dis = face_recognition.face_distance(encodeListKnown,encode_face)

        matchIndex = np.argmin(face_dis)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            y1,x2,y2,x1 = face_loc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2+35),(x2,y2),(0,255,0), cv2.FILLED)
            cv2.putText(img ,name, (x1+6,y2+30), cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,255),2)
            markAttendance(name)

    cv2.imshow("Webcam",img)
    key = cv2.waitKey(1)
    if key ==27:
        break
cap.release()
cv2.destroyAllWindows()