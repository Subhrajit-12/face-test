import json
import os
import time
from datetime import date

from flask import Flask, render_template, Response, request, jsonify, redirect, session
import cv2
from PIL import Image
import numpy as np
import pickle
import requests
from passlib.hash import sha512_crypt
from requests.auth import HTTPBasicAuth
from datalayer import update_by_account_number, retrieve_by_account_number

app = Flask(__name__)
app.config['SECRET_KEY'] = 'abcd'

camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)

            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def take_photo(name):
    gen_frames()
    os.makedirs('images/' + name)
    for i in range(10):
        time.sleep(1)
        return_value, image = camera.read()

        cv2.imwrite('images/' + name + '/opencv' + str(i) + '.png', image)


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        butt = request.form.get('but')
        fname = request.form.get('fname')
        dob = request.form.get('dob')
        a = dob.split('-')
        age = calculateAge(date(int(a[0]), int(a[1]), int(a[2])))
        gender = 'M' if request.form.get('gender') == "Male" else 'F'
        mnumber = request.form.get('mnumber')
        password = request.form.get('password')
        if butt == 'Click Me':
            take_photo(fname)
            paras = {
                "name": fname,
                "age": age,
                "gender": gender,
                "password": sha512_crypt.encrypt(password),
                "mobile": mnumber,
            }
            print(json.dumps(paras, indent=4))
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*"
            }
            response = requests.post(url="http://127.0.0.1:8000/users/", data=json.dumps(paras, indent=4),
                                     headers=headers,
                                     auth=HTTPBasicAuth('Raj', 'RajRaj@7'))
            if response.status_code.__eq__(201):
                train_face()
                return redirect('landing_page')
            else:
                print("Something went wrong")

            # face_recog.run_con()

    return render_template("index.html")


@app.route("/content", methods=['GET', 'POST'])
def content():
    anumber = session.get('anumber', None)
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*"
    }
    response = requests.get(url=f'http://127.0.0.1:8000/users/{anumber}',
                            headers=headers,
                            auth=HTTPBasicAuth('Raj', 'RajRaj@7'))
    res_body = response.json()
    data = retrieve_by_account_number(anumber)
    balance=[d[3] for d in data]

    content = {
        "name": res_body['name'],
        "id": res_body['id'],
        "mobile": res_body['mobile'],
        "Balance": balance[0]
    }

    if request.method == 'POST':
        amount = request.form.get('tobe')
        butt = request.form.get('but-withdraw')
        if butt == 'withdraw':
            new_amount = int(balance[0]) - int(amount)
            update_by_account_number(anumber, str(new_amount))
            return redirect('content')

    return render_template('content.html', content=content)


@app.route('/profile-balance')
def withdraw():
    anumber = session.get('anumber', None)
    update_by_account_number()


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        butt = request.form.get('but')
        fname = request.form.get('fname')
        anumber = request.form.get('anumber')
        password = request.form.get('password')
        if butt == 'Click Me':
            if is_verified(fname, anumber, password):
                session['fname'] = fname
                session['anumber'] = anumber
                return redirect('face_login')
    return render_template('login.html')


@app.route("/face_login", methods=['GET', 'POST'])
def face_login():
    if request.method == 'POST':
        butt = request.form.get('but')
        if butt == 'Click Me':
            fname = session.get('fname', None)
            if check_face(fname):
                return redirect('content')
            else:
                print('Not Matched')
    return render_template('face_login.html')


@app.route("/health")
def health():
    return jsonify(status='UP')


def train_face():
    image_dir = r'E:\PythonBasics\pythonProject\face-test-11-new-16-mar\images'
    face_cascade = cv2.CascadeClassifier('face_recog/cascades/data/haarcascade_frontalface_alt2.xml')

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
                faces = face_cascade.detectMultiScale(image_array, minNeighbors=7)

                for x, y, w, h in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('trainer.yml')


def check_face(username):
    face_cascade = cv2.CascadeClassifier('face_recog/cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    labels = {}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    gen_frames()
    ret, frame = camera.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray_img[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if 27 <= conf <= 85:
            print(conf)
            print(id_)
            print(labels[id_])
            if string_format(username).__eq__(labels[id_]):
                return True
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


@app.route('/')
@app.route("/landing_page")
def landing_page():
    return render_template('landingPage.html')


def calculateAge(birthDate):
    today = date.today()
    age = today.year - birthDate.year - ((today.month, today.day) < (birthDate.month, birthDate.day))
    return age


def string_format(name):
    return name.replace(" ", "-").lower()


def is_verified(fname, anumber, password):
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*"
    }
    response = requests.get(url=f'http://127.0.0.1:8000/users/{anumber}',
                            headers=headers,
                            auth=HTTPBasicAuth('Raj', 'RajRaj@7'))
    res_body = response.json()
    if res_body['name'] == fname and sha512_crypt.verify(password, res_body['password']):
        return True
    else:
        return False


if __name__ == "__main__":
    app.run(debug=True)
