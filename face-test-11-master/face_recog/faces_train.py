import os
from PIL import Image
import numpy as np
import cv2
import pickle


def run_con():
    image_dir = r'E:\PythonBasics\pythonProject\face-test-11-new-16-mar\images'
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    x_train = []
    y_labels = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(' ', '-').lower()
                # y_labels.append(label)
                # x_train.append(path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                pil_image = Image.open(path).convert("L")  # grayscale
                image_array = np.array(pil_image, "uint8")
                faces = face_cascade.detectMultiScale(image_array, minNeighbors=5)

                for x, y, w, h in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('trainer.yml')

run_con()
