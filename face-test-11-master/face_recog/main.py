import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray_img[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 85:
            print(conf)
            print(id_)
            print(labels[id_])
        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        encord_x = x + w
        encord_y = y + h
        cv2.rectangle(frame, (x, y), (encord_x, encord_y), color, stroke)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
