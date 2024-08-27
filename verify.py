import cv2
import numpy as np
import pyrealsense2 as rs
import face_recognition
import dlib
import os

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load the facial landmark predictor
shape_predictor_path = 'data/shape_predictor/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)
detector = dlib.get_frontal_face_detector()

# Directories to load face encodings and landmarks
encoding_folder = 'data/encodings/'
landmarks_folder = 'data/landmarks/'

def load_face_encodings_and_names(folder):
    """Load all face encodings and corresponding names from the specified folder."""
    encodings = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            path = os.path.join(folder, filename)
            encoding = np.load(path)
            encodings.append(encoding)
            name = os.path.splitext(filename)[0]  # Assuming file name is the person's name
            names.append(name)
    return encodings, names

def load_face_landmarks(folder):
    """Load all facial landmarks from the specified folder."""
    landmarks = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            path = os.path.join(folder, filename)
            landmark = np.load(path)
            landmarks.append(landmark)
    return landmarks

def detect_faces(image):
    """Detect faces in an image using a Haar cascade classifier."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 4)

def get_face_encoding(face_image):
    """Get the face encoding for a given face image."""
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(face_rgb)
    return face_encodings[0] if face_encodings else None

def get_face_landmarks(gray, rect):
    """Get the facial landmarks for a detected face."""
    shape = predictor(gray, rect)
    return np.array([(p.x, p.y) for p in shape.parts()])

def verify_face():
    """Verify faces in real-time and provide feedback."""
    known_encodings, known_names = load_face_encodings_and_names(encoding_folder)
    known_landmarks = load_face_landmarks(landmarks_folder)
    
    print("Starting verification...")
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        faces = detect_faces(color_image)
        for (x, y, w, h) in faces:
            face_image = color_image[y:y+h, x:x+w]
            depth = depth_frame.get_distance(x + w//2, y + h//2)
            face_encoding = get_face_encoding(face_image)
            rect = dlib.rectangle(x, y, x+w, y+h)
            face_landmarks = get_face_landmarks(gray, rect)
            
            if face_encoding is not None:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    print(f"Face authenticated: {name}")
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Verified: {name}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Depth: {depth:.2f}m", (x, y+h+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Draw facial landmarks
                    for (lx, ly) in face_landmarks:
                        cv2.circle(color_image, (lx, ly), 2, (255, 0, 0), -1)
                else:
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(color_image, "Not recognized", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    verify_face()
    pipeline.stop()
    cv2.destroyAllWindows()
