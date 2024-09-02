from dependencies import *

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load the facial landmark predictor
shape_predictor_path = '/home/inlab22/auth/data/shape_predictor/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)
detector = dlib.get_frontal_face_detector()

# Directory to save face encodings and landmarks
encoding_folder = 'data/encodings/'
landmarks_folder = 'data/landmarks/'
os.makedirs(encoding_folder, exist_ok=True)
os.makedirs(landmarks_folder, exist_ok=True)

def save_face_encoding(face_encoding, filename):
    """Save the face encoding to a file."""
    try:
        np.save(filename, face_encoding)
        print(f"Face encoding saved to {filename}.")
    except Exception as e:
        print(f"Error saving face encoding: {e}")

def save_landmarks(landmarks, filename):
    """Save the facial landmarks to a file."""
    try:
        np.save(filename, landmarks)
        print(f"Facial landmarks saved to {filename}.")
    except Exception as e:
        print(f"Error saving facial landmarks: {e}")

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
    name = input("Enter the name of the person being enrolled: ").strip()
    if not name:
        print("Name cannot be empty.")
        return
    
    print("Press 's' to capture and save the face encoding and landmarks.")
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("No frame data available, skipping.")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        faces = detect_faces(color_image)
        if len(faces) == 0:
            print("No face detected.")
            continue

        for (x, y, w, h) in faces:
            face_image = color_image[y:y+h, x:x+w]
            face_encoding = get_face_encoding(face_image)
            rect = dlib.rectangle(x, y, x+w, y+h)
            face_landmarks = get_face_landmarks(gray, rect)
            
            # Check depth at the center of the detected face
            depth = depth_frame.get_distance(x + w // 2, y + h // 2)
            print(f"Depth at face center: {depth:.2f} meters")

            # Ensure the detected face is within a reasonable depth range (e.g., 0.3m to 1.5m)
            if 0 <= depth <= 10:
                if face_encoding is not None:
                    encoding_filename = os.path.join(encoding_folder, f'{name}.npy')
                    save_face_encoding(face_encoding, encoding_filename)
                    
                    landmarks_filename = os.path.join(landmarks_folder, f'{name}_landmarks.npy')
                    save_landmarks(face_landmarks, landmarks_filename)
                    
                    print(f"{name} has been successfully enrolled.")
                    break  # Exit the loop after saving the face
                else:
                    print("Face encoding could not be obtained.")
            else:
                print("No valid face detected within the required depth range.")

        cv2.imshow('Enroll Face', color_image)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

if __name__ == "__main__":
    enroll_face()
    pipeline.stop()
    cv2.destroyAllWindows()
