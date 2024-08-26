# utils.py
import cv2
import dlib
import face_recognition
import numpy as np
from imutils import face_utils

# Load dlib's detector and predictor
p = "data/shape_predictor/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def get_face_encodings(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face_image = image[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(face_rgb)
        
        if len(face_encoding) > 0:
            return face_encoding[0], image

    return None, image

def compare_faces(face_encoding, known_encodings, threshold=0.6):
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    
    if face_distances[best_match_index] < threshold:
        return True, "Authenticated"
    else:
        return False, "Unknown"
