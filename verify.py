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

# Directory to load face encodings
encoding_folder = 'data/known_faces/'

def load_face_encodings(folder):
    """Load all face encodings from the specified folder."""
    encodings = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            path = os.path.join(folder, filename)
            encoding = np.load(path)
            encodings.append(encoding)
    return encodings

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

def verify_face():
    """Verify faces in real-time and provide feedback."""
    known_encodings = load_face_encodings(encoding_folder)
    
    print("Starting verification...")
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        
        faces = detect_faces(color_image)
        for (x, y, w, h) in faces:
            face_image = color_image[y:y+h, x:x+w]
            depth = depth_frame.get_distance(x + w//2, y + h//2)
            face_encoding = get_face_encoding(face_image)
            
            if face_encoding is not None:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                if True in matches:
                    print("Face authenticated")
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Verified", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(color_image, f"Depth: {depth:.2f}m", (x, y+h+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
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
