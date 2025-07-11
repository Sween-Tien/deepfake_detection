import cv2
import dlib
import numpy as np
from scipy.fftpack import dct

predictor = dlib.shape_predictor("models\pretrained\dlib\shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def extract_geometric_featuress(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return np.zeros(68 * 2)
    shape = predictor(gray, faces[0])
    coords = np.array([[pt.x, pt.y] for pt in shape.parts()])
    return coords.flatten()

def extract_dct_features(gray_image, top_k=64):
    gray = cv2.resize(gray_image, (128, 128))
    dct_trans = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    flat = dct_trans.flatten()
    return flat[:top_k]
