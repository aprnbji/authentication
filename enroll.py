import cv2
import numpy as np
import pyrealsense2 as rs
import face_recognition
import os

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Directory to save face encodings
encoding_folder = 'data/known_faces/'
os.makedirs(encoding_folder, exist_ok=True)

def save_face_encoding(face_encoding, filename):
    """Save the face encoding to a file."""
    np.save(filename, face_encoding)
    print(f"Face encoding saved to {filename}.")

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

def enroll_face():
    """Enroll a face and save its encoding."""
    print("Press 's' to capture and save a face encoding.")
    count = 1
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        faces = detect_faces(color_image)
        for (x, y, w, h) in faces:
            face_image = color_image[y:y+h, x:x+w]
            face_encoding = get_face_encoding(face_image)
            
            if face_encoding is not None:
                filename = os.path.join(encoding_folder, f'face_{count}.npy')
                save_face_encoding(face_encoding, filename)
                count += 1
        
        cv2.imshow('Enroll Face', color_image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

if __name__ == "__main__":
    enroll_face()
    pipeline.stop()
    cv2.destroyAllWindows()
