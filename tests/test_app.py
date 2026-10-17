"""Self-checking tests for app.py.

No camera is needed: the faces are synthetic images (a dark user and a bright user),
so the expected result of every check can be worked out by hand.
Each test runs in an empty temporary folder, so the real Attendance/ and static/
folders are never touched.
"""
import os
from datetime import date as real_date, datetime as real_datetime

import cv2
import numpy as np
import pytest

import app


def check(obtained, expected, what):
    assert obtained == expected, f"ECHEC {what}: obtenu {obtained!r}, attendu {expected!r}"


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, 'UNKNOWN_THRESHOLD', None)
    return tmp_path


@pytest.fixture
def fixed_clock(monkeypatch):
    # 15 January 2023 at 08:30:00, the same date as the example in my report
    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return real_date(2023, 1, 15)

    class FakeDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2023, 1, 15, 8, 30, 0)

    monkeypatch.setattr(app, 'date', FakeDate)
    monkeypatch.setattr(app, 'datetime', FakeDateTime)


def make_user(label, base, nb=50, seed=0):
    """Saves nb noisy 80x80 face images of mean value `base` in static/faces/<label>/."""
    rng = np.random.default_rng(seed)
    os.makedirs(f'static/faces/{label}')
    for k in range(nb):
        img = np.clip(base + rng.integers(-20, 20, (80, 80, 3)), 0, 255).astype(np.uint8)
        cv2.imwrite(f'static/faces/{label}/{k}.jpg', img)


def flat_face(value):
    """A uniform 50x50 colour face, flattened like in start(): 50*50*3 = 7500 values."""
    return np.full((50, 50, 3), value, np.uint8).reshape(1, -1)


# ---------- date and file names ----------

def test_date_formats(fixed_clock):
    check(app.datetoday(), '01_15_23', 'datetoday')
    check(app.datetoday2(), '15-January-2023', 'datetoday2')
    check(app.attendance_file(), 'Attendance/Attendance-01_15_23.csv', 'attendance_file')


def test_init_folders_creates_empty_sheet(fixed_clock):
    app.init_folders()
    for folder in ['Attendance', 'static', 'static/faces']:
        check(os.path.isdir(folder), True, f'dossier {folder}')
    check(open(app.attendance_file()).read(), 'Name,Roll,Time', 'en-tete du CSV')
    check(app.totalreg(), 0, 'totalreg')


# ---------- face detection ----------

def test_extract_faces_without_image():
    # the original "img != []" crashed here with recent numpy
    check(len(app.extract_faces(None)), 0, 'extract_faces(None)')
    check(len(app.extract_faces(np.zeros((0, 0, 3), np.uint8))), 0, 'extract_faces(image vide)')


def test_extract_faces_black_frame():
    black = np.zeros((480, 640, 3), np.uint8)
    check(len(app.extract_faces(black)), 0, 'visages sur une image noire')


# ---------- attendance sheet ----------

def test_add_attendance(fixed_clock):
    app.init_folders()
    check(app.add_attendance('amine_1'), True, 'premier pointage amine_1')
    check(app.add_attendance('amine_1'), False, 'deuxieme pointage amine_1 (doublon)')
    check(app.add_attendance('Unknown'), False, 'pointage Unknown')
    check(app.add_attendance('sara_2'), True, 'premier pointage sara_2')
    check(open(app.attendance_file()).read(),
          'Name,Roll,Time\namine,1,08:30:00\nsara,2,08:30:00', 'contenu du CSV')

    names, rolls, times, l = app.extract_attendance()
    check(l, 2, 'nombre de lignes')
    check(list(names), ['amine', 'sara'], 'noms')
    check(list(rolls), [1, 2], 'IDs')


# ---------- training and recognition ----------

def test_train_model_without_faces():
    app.init_folders()
    check(app.train_model(), False, 'train_model sans aucune image')


def test_train_and_identify():
    app.init_folders()
    make_user('amine_1', base=40, seed=1)   # dark faces, values 20..59
    make_user('sara_2', base=200, seed=2)   # bright faces, values 180..219
    check(app.train_model(), True, 'train_model')
    check(app.totalreg(), 2, 'totalreg')
    check(app.identify_face(flat_face(45)), 'amine_1', 'visage sombre (45)')
    check(app.identify_face(flat_face(210)), 'sara_2', 'visage clair (210)')


def test_identify_then_add_attendance(fixed_clock):
    # regression test: in the report version identify_face()[0] gave "a" and add_attendance crashed
    app.init_folders()
    make_user('amine_1', base=40, seed=1)
    make_user('sara_2', base=200, seed=2)
    app.train_model()
    person = app.identify_face(flat_face(45))
    check(app.add_attendance(person), True, 'pointage apres reconnaissance')
    check(open(app.attendance_file()).read(), 'Name,Roll,Time\namine,1,08:30:00', 'contenu du CSV')


def test_fewer_than_five_images():
    # Esc pressed after 3 images: k must drop to 3, otherwise predict() fails
    app.init_folders()
    make_user('amine_1', base=40, nb=3, seed=1)
    check(app.train_model(), True, 'train_model avec 3 images')
    check(app.identify_face(flat_face(45)), 'amine_1', 'reconnaissance avec 3 images')


def test_unknown_threshold(monkeypatch):
    app.init_folders()
    make_user('amine_1', base=40, seed=1)
    make_user('sara_2', base=200, seed=2)
    app.train_model()
    # without threshold a KNN always picks someone, even for a grey face (128)
    check(app.identify_face(flat_face(128)), 'sara_2', 'visage gris sans seuil')
    # distance ~ sqrt(7500) * |mean difference| = 86.6 * delta
    # grey face: 86.6 * |128 - 200| = 6235 from sara (measured 6189), dark face (45): about 700 from amine
    # so a threshold of 2000 rejects the grey face and keeps the dark one
    monkeypatch.setattr(app, 'UNKNOWN_THRESHOLD', 2000.0)
    check(app.identify_face(flat_face(128)), 'Unknown', 'visage gris avec seuil')
    check(app.identify_face(flat_face(45)), 'amine_1', 'visage sombre avec seuil')


def test_deleted_user_is_unknown():
    app.init_folders()
    make_user('amine_1', base=40, seed=1)
    app.train_model()
    os.rename('static/faces/amine_1', 'static/faces_removed')
    check(app.identify_face(flat_face(45)), 'Unknown', 'utilisateur supprime apres entrainement')


# ---------- Flask routes ----------

def test_home_page():
    client = app.app.test_client()
    page = client.get('/')
    check(page.status_code, 200, 'code HTTP de /')
    check(b'Total Users in Database: 0' in page.data, True, 'compteur utilisateurs')


def test_start_without_model():
    page = app.app.test_client().get('/start')
    check(b'There is no trained model' in page.data, True, 'message sans modele')


@pytest.mark.parametrize('name, user_id', [('amine_x', '1'), ('amine', 'abc'), ('', '1')])
def test_add_rejects_bad_form(name, user_id):
    page = app.app.test_client().post('/add', data={'newusername': name, 'newuserid': user_id})
    check(b'must not contain' in page.data, True, f'formulaire refuse ({name!r}, {user_id!r})')
    check(os.listdir('static/faces'), [], 'aucun dossier cree')
