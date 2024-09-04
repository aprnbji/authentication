from dependencies import *

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# Directory to save known face encodings and landmarks
ENCODING_DIR = 'data/encodings'
LANDMARKS_DIR = 'data/landmarks'
os.makedirs(ENCODING_DIR, exist_ok=True)
os.makedirs(LANDMARKS_DIR, exist_ok=True)

def save_face_encoding(name, face_encoding):
    """Save the face encoding to a file."""
    filename = os.path.join(ENCODING_DIR, f'{name}.npy')
    try:
        np.save(filename, face_encoding)
        print(f"Face encoding saved to {filename}.")
    except Exception as e:
        print(f"Error saving face encoding: {e}")

def save_landmarks(name, landmarks):
    """Save the facial landmarks to a file."""
    filename = os.path.join(LANDMARKS_DIR, f'{name}.npy')
    try:
        np.save(filename, landmarks)
        print(f"Facial landmarks saved to {filename}.")
    except Exception as e:
        print(f"Error saving facial landmarks: {e}")

def detect_faces(image):
    """Detect faces in an image using MediaPipe Face Mesh."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    return results

def calculate_depth_variation(face_landmarks, depth_frame, color_image_shape):
    """Calculate the average depth and variation across the face region."""
    depth_values = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * color_image_shape[1])
        y = int(lm.y * color_image_shape[0])
        depth = depth_frame.get_distance(x, y)
        depth_values.append(depth)
    
    avg_depth = np.mean(depth_values)
    depth_variation = np.std(depth_values)
    
    return avg_depth, depth_variation

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

        # Detect faces using MediaPipe
        results = detect_faces(color_image)
        if not results.multi_face_landmarks:
            print("No face detected.")
            continue

        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=color_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            mp_drawing.draw_landmarks(
                image=color_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            mp_drawing.draw_landmarks(
                image=color_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

            # Calculate the average depth and depth variation of the face region
            avg_depth, depth_variation = calculate_depth_variation(face_landmarks, depth_frame, color_image.shape)
            print(f"Average depth of face: {avg_depth:.2f} meters")
            print(f"Depth variation: {depth_variation:.4f} meters")

            # Ensure the detected face is within a reasonable depth range and variation
            if avg_depth == 0 or depth_variation < 0.01:  # Adjust the threshold as needed
                print("Depth variation too low. Likely a flat image. Cannot enroll.")

                # Draw a bounding box around the image and display the message
                height, width, _ = color_image.shape
                cv2.rectangle(color_image, (10, 10), (width - 10, height - 10), (0, 0, 255), 2)
                cv2.putText(color_image, "Images cannot be enrolled", 
                            (50, height - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                continue

            if 0 < avg_depth <= 7:
                face_image = color_image
                face_encoding = face_recognition.face_encodings(face_image)[0] if face_recognition.face_encodings(face_image) else None
                
                if face_encoding is not None:
                    save_face_encoding(name, face_encoding)
                    
                    face_landmarks_array = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                    save_landmarks(name, face_landmarks_array)
                    
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
