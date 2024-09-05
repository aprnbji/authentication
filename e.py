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

# Directory to save face encodings and landmarks
encoding_folder = 'data/encodings/'
landmarks_folder = 'data/landmarks/'
os.makedirs(encoding_folder, exist_ok=True)
os.makedirs(landmarks_folder, exist_ok=True)

def save_face_encoding(face_encoding, filename):
    """Save the face encoding to a file."""
    np.save(filename, face_encoding)
    print(f"Face encoding saved to {filename}.")

def save_landmarks(landmarks, filename):
    """Save the facial landmarks to a file."""
    np.save(filename, landmarks)
    print(f"Facial landmarks saved to {filename}.")

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

def calculate_depth_statistics(depth_frame, rect):
    """Calculate depth statistics for a detected face."""
    depth_image = np.asanyarray(depth_frame.get_data())
    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    face_depths = [depth_frame.get_distance(x + i, y + j) for i in range(w) for j in range(h) if depth_frame.get_distance(x + i, y + j) > 0]
    
    if face_depths:
        avg_depth = np.mean(face_depths)
        depth_variation = np.std(face_depths)
        return avg_depth, depth_variation
    else:
        return None, None

def enroll_face():
    """Enroll a face and save its encoding and landmarks."""
    person_name = input("Enter the name of the person being enrolled: ").strip()
    print("Press 's' to capture and save a face encoding and landmarks.")
    
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        faces = detect_faces(color_image)
        for (x, y, w, h) in faces:
            face_image = color_image[y:y+h, x:x+w]
            face_encoding = get_face_encoding(face_image)
            rect = dlib.rectangle(x, y, x+w, y+h)
            face_landmarks = get_face_landmarks(gray, rect)
            
            avg_depth, depth_variation = calculate_depth_statistics(depth_frame, rect)
            
            if face_encoding is not None and avg_depth is not None:
                if depth_variation < 0.01:  # Threshold for differentiating between 2D and 3D objects
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(color_image, "Images can't be enrolled", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    encoding_filename = os.path.join(encoding_folder, f'{person_name}_encoding.npy')
                    save_face_encoding(face_encoding, encoding_filename)
                    
                    landmarks_filename = os.path.join(landmarks_folder, f'{person_name}_landmarks.npy')
                    save_landmarks(face_landmarks, landmarks_filename)
                
        cv2.imshow('Enroll Face', color_image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

if __name__ == "__main__":
    enroll_face()
    pipeline.stop()
    cv2.destroyAllWindows()
