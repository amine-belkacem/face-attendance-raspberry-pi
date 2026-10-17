import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

#### Defining Flask App
app = Flask(__name__)

#### Settings
# Camera index: 0 = first webcam found. On my setup the USB webcam was the only camera on the Pi.
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', 0))
# Max distance to the nearest stored face before a face is called "Unknown".
# A KNN always answers with one of the known users, so without this check a stranger
# is recorded as someone else. Not tuned yet (no measurements), so it is off by default.
UNKNOWN_THRESHOLD = float(os.environ['UNKNOWN_THRESHOLD']) if 'UNKNOWN_THRESHOLD' in os.environ else None
NB_IMAGES = 50  # face images saved per new user


#### Today's date in 2 different formats
# Computed on each call (and not once at startup) so a kiosk left running overnight
# writes to the right file the next morning.
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


def attendance_file():
    return f'Attendance/Attendance-{datetoday()}.csv'


#### Face detector (Haar cascade shipped with OpenCV, no need to copy the XML file)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#### If these directories don't exist, create them
def init_folders():
    if not os.path.isdir('Attendance'):
        os.makedirs('Attendance')
    if not os.path.isdir('static'):
        os.makedirs('static')
    if not os.path.isdir('static/faces'):
        os.makedirs('static/faces')
    if not os.path.isfile(attendance_file()):
        with open(attendance_file(), 'w') as f:
            f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    # "img != []" crashes with recent numpy when img is an image, so test for None / empty instead
    if img is not None and img.size > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []


#### Identify face using ML model
# Returns the full label "name_id" (needed by add_attendance) or "Unknown".
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    if UNKNOWN_THRESHOLD is not None:
        distance, _ = model.kneighbors(facearray, n_neighbors=1)
        if distance[0][0] > UNKNOWN_THRESHOLD:
            return "Unknown"  # too far from every registered face
    label = model.predict(facearray)[0]

    registered_users = os.listdir('static/faces')
    if label in registered_users:
        return label
    else:
        return "Unknown"  # the user folder was deleted after training


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    if len(faces) == 0:
        return False
    faces = np.array(faces)
    # 5 neighbours, unless Esc was pressed early and fewer than 5 images exist
    knn = KNeighborsClassifier(n_neighbors=min(5, len(faces)))
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    return True


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(attendance_file())
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


#### Add Attendance of a specific user
# Only the first detection of the day is kept (arrival time).
def add_attendance(name):
    if name == "Unknown":
        return False
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file())
    if int(userid) not in list(df['Roll']):
        with open(attendance_file(), 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        return True
    return False


def render_home(mess=None):
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2(), mess=mess)


#### Our main page
@app.route('/')
def home():
    init_folders()
    return render_home()


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    init_folders()
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_home('There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # camera unplugged or not found
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person.split("_")[0]}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_home()


#### This function will run when we add a new user
@app.route('/add', methods=['POST'])
def add():
    init_folders()
    newusername = request.form['newusername'].strip()
    newuserid = request.form['newuserid'].strip()
    # the name and the ID are stored as "name_id", so "_" in the name would break the split
    if not newusername or '_' in newusername or not newuserid.isdigit():
        return render_home('The name must not contain "_" and the ID must be a number.')
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    i, j = 0, 0
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{NB_IMAGES}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j >= NB_IMAGES*10:  # ">=": with 2 faces in a frame j can jump over 500
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    if not train_model():
        return render_home('No face was captured, the model was not trained.')
    return render_home()


#### Our main function which runs the Flask App
if __name__ == '__main__':
    init_folders()
    app.run(debug=True)
