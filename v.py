from dependencies import *

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

context = rs.context()

# List all connected devices
devices = context.query_devices()
for i, device in enumerate(devices):
    print(f"Device {i}: {device.get_info(rs.camera_info.name)}")

camera_index = 0  # Change this index to select a different camera
selected_device = devices[camera_index]

# Configure the pipeline to use the selected device
config.enable_device(selected_device.get_info(rs.camera_info.serial_number))

# Start streaming
pipeline.start(config)

# Load the facial landmark predictor
shape_predictor_path = '/home/inlab22/auth/data/shape_predictor/shape_predictor_68_face_landmarks.dat'
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
    try:
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
                # Check depth at the center of the detected face
                depth = depth_frame.get_distance(x + w // 2, y + h // 2)
                print(f"Depth at face center: {depth:.2f} meters")

                # Ensure the detected face is within a reasonable depth range (e.g., 0.3m to 1.5m)
                if 0 < depth <= 10:
                    face_image = color_image[y:y+h, x:x+w]
                    face_encoding = get_face_encoding(face_image)
                    rect = dlib.rectangle(x, y, x+w, y+h)
                    face_landmarks = get_face_landmarks(gray, rect)
                    
                    if face_encoding is not None:
                        matches = face_recognition.compare_faces(known_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            status_text = f"Verified: {name}"
                            box_color = (0, 255, 0)
                            depth_text = f"Depth: {depth:.2f}m"
                        else:
                            status_text = "Not recognized"
                            box_color = (0, 0, 255)
                            depth_text = f"Depth: {depth:.2f}m"

                        print(f"Face authenticated: {name}, Depth: {depth:.2f}m")
                        cv2.rectangle(color_image, (x, y), (x+w, y+h), box_color, 2)
                        cv2.putText(color_image, status_text, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                        cv2.putText(color_image, depth_text, (x, y+h+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                        
                        # Draw facial landmarks
                        for (lx, ly) in face_landmarks:
                            cv2.circle(color_image, (lx, ly), 2, (255, 0, 0), -1)
                    else:
                        print("Face encoding could not be obtained.")
                elif depth == 0:
                    cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(color_image, "Not authenticated", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    print("Face detected but not within depth range.")

            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Ensure the pipeline is properly stopped when done
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_face()
