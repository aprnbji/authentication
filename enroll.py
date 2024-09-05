from dependencies import *

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

def enroll_face():
    """Enroll a face and save its encoding and landmarks."""
    person_name = input("Enter the name of the person being enrolled: ").strip()
    print("Press 's' to capture and save a face encoding and landmarks.")
    
    count = 1
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        faces = detect_faces(color_image)
        for (x, y, w, h) in faces:
            face_image = color_image[y:y+h, x:x+w]
            face_encoding = get_face_encoding(face_image)
            rect = dlib.rectangle(x, y, x+w, y+h)
            face_landmarks = get_face_landmarks(gray, rect)
            
            if face_encoding is not None:
                encoding_filename = os.path.join(encoding_folder, f'{person_name}_encoding.npy')
                save_face_encoding(face_encoding, encoding_filename)
                
                landmarks_filename = os.path.join(landmarks_folder, f'{person_name}_landmarks.npy')
                save_landmarks(face_landmarks, landmarks_filename)
                
                count += 1
        
        cv2.imshow('Enroll Face', color_image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

if __name__ == "__main__":
    enroll_face()
    pipeline.stop()
    cv2.destroyAllWindows()
