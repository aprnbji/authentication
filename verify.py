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

# Directory for saved landmarks and encodings
ENCODING_DIR = 'data/encodings/'
LANDMARKS_DIR = 'data/landmarks/'

def load_face_encodings(folder):
    """Load all face encodings and corresponding names from the specified folder."""
    encodings = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            path = os.path.join(folder, filename)
            encoding = np.load(path)
            encodings.append(encoding)
            name = os.path.splitext(filename)[0]
            names.append(name)
    return encodings, names

def load_face_landmarks(folder):
    """Load all facial landmarks from the specified folder."""
    landmarks = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            path = os.path.join(folder, filename)
            landmark = np.load(path)
            landmarks.append(landmark)
            name = os.path.splitext(filename)[0]
            names.append(name)
    return landmarks, names

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

def verify_face():
    """Verify faces in real-time and provide feedback."""
    known_encodings, known_names = load_face_encodings(ENCODING_DIR)
    known_landmarks, known_landmark_names = load_face_landmarks(LANDMARKS_DIR)
    
    print("Starting verification...")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            
            results = detect_faces(color_image)
            if not results.multi_face_landmarks:
                print("No face detected.")
                continue

            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image=color_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Calculate average depth and depth variation
                avg_depth, depth_variation = calculate_depth_variation(face_landmarks, depth_frame, color_image.shape)
                print(f"Average depth: {avg_depth:.2f} meters, Depth variation: {depth_variation:.4f} meters")

                # Check if depth variation is too low
                if avg_depth == 0 or depth_variation < 0.07:
                    print("Depth variation too low. Likely a flat image. Cannot verify.")
                    # Draw bounding box and display the message
                    height, width, _ = color_image.shape
                    cv2.rectangle(color_image, (10, 10), (width - 10, height - 10), (0, 0, 255), 2)
                    cv2.putText(color_image, "Images can't be authenticated", 
                                (50, height - 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                    continue

                if 0 < avg_depth <= 7:
                    face_image = color_image
                    face_encoding = face_recognition.face_encodings(face_image)[0] if face_recognition.face_encodings(face_image) else None
                    
                    if face_encoding is not None:
                        matches = face_recognition.compare_faces(known_encodings, face_encoding)
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            status_text = f"Verified: {name}"
                            box_color = (0, 255, 0)
                        else:
                            status_text = "Not recognized"
                            box_color = (0, 0, 255)

                        print(f"Face authenticated by encoding: {name}, Depth: {avg_depth:.2f}m")
                    else:
                        # Fallback to landmarks if encoding is not available
                        min_distance = float('inf')
                        matched_name = "Not recognized"
                        
                        for i, known_landmark in enumerate(known_landmarks):
                            face_landmarks_array = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                            if face_landmarks_array.shape == known_landmark.shape:
                                dist = np.linalg.norm(face_landmarks_array - known_landmark)
                                if dist < min_distance:
                                    min_distance = dist
                                    matched_name = known_landmark_names[i]

                        threshold = 0.05  # Adjust this threshold as needed
                        if min_distance < threshold:
                            status_text = f"Verified: {matched_name}"
                            box_color = (0, 255, 0)
                        else:
                            status_text = "Not recognized"
                            box_color = (0, 0, 255)

                        print(f"Face verified by landmarks: {matched_name}, Depth: {avg_depth:.2f}m")

                    # Draw bounding box and status text
                    x_min = int(min([lm.x for lm in face_landmarks.landmark]) * color_image.shape[1])
                    y_min = int(min([lm.y for lm in face_landmarks.landmark]) * color_image.shape[0])
                    x_max = int(max([lm.x for lm in face_landmarks.landmark]) * color_image.shape[1])
                    y_max = int(max([lm.y for lm in face_landmarks.landmark]) * color_image.shape[0])
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), box_color, 2)
                    cv2.putText(color_image, status_text, (x_min, y_min-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                else:
                    print("Face detected but not within the acceptable depth range.")

            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Ensure the pipeline is properly stopped when done
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_face()
