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
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Directory for saved landmarks and encodings
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
            name = os.path.splitext(filename)[0]
            names.append(name)
    return encodings, names

def load_face_landmarks(folder):
    """Load all facial landmarks from the specified folder."""
    landmarks = []
    names = []
    for filename in os.listdir(folder):
        if filename.endswith('_landmarks.npy'):
            path = os.path.join(folder, filename)
            landmark = np.load(path)
            landmarks.append(landmark)
            name = os.path.splitext(filename)[0].replace('_landmarks', '')
            names.append(name)
    return landmarks, names

def detect_faces(image):
    """Detect faces in an image using MediaPipe Face Mesh."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    return results

def get_face_encoding(face_image):
    """Get the face encoding for a given face image."""
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(face_rgb)
    return face_encodings[0] if face_encodings else None

def get_face_landmarks(results):
    """Get the facial landmarks from MediaPipe results."""
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks.extend([(lm.x, lm.y) for lm in face_landmarks.landmark])
    return np.array(landmarks)

def calculate_landmark_distance(landmarks1, landmarks2):
    """Calculate the Euclidean distance between two sets of landmarks."""
    return np.linalg.norm(landmarks1 - landmarks2)

def make_square_bounding_box(x_min, y_min, x_max, y_max):
    """Adjust bounding box to be a square."""
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height)

    x_center = x_min + width // 2
    y_center = y_min + height // 2

    half_size = size // 2
    x_min_square = x_center - half_size
    y_min_square = y_center - half_size
    x_max_square = x_center + half_size
    y_max_square = y_center + half_size

    return x_min_square, y_min_square, x_max_square, y_max_square

def verify_face():
    """Verify faces in real-time and provide feedback."""
    known_encodings, known_names = load_face_encodings_and_names(encoding_folder)
    known_landmarks, known_landmark_names = load_face_landmarks(landmarks_folder)
    
    print("Starting verification...")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            results = detect_faces(color_image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image=color_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    # Calculate face bounding box
                    bbox = face_landmarks.landmark
                    x_min = int(min([lm.x for lm in bbox]) * color_image.shape[1])
                    x_max = int(max([lm.x for lm in bbox]) * color_image.shape[1])
                    y_min = int(min([lm.y for lm in bbox]) * color_image.shape[0])
                    y_max = int(max([lm.y for lm in bbox]) * color_image.shape[0])

                    # Make the bounding box square
                    x_min, y_min, x_max, y_max = make_square_bounding_box(x_min, y_min, x_max, y_max)

                    # Check depth at each landmark and calculate the average depth
                    depths = []
                    for lm in face_landmarks.landmark:
                        x = int(lm.x * color_image.shape[1])
                        y = int(lm.y * color_image.shape[0])
                        depth = depth_image[y, x]
                        if depth > 0:  # Ensure valid depth
                            depths.append(depth)
                    
                    if depths:
                        avg_depth = np.mean(depths) / 1000.0  # Convert from mm to meters
                        print(f"Average depth of face: {avg_depth:.2f} meters")

                        # Ensure the detected face is within a reasonable depth range
                        if 0 < avg_depth <= 1.0:  # Assuming depth in meters
                            face_image = color_image[y_min:y_max, x_min:x_max]
                            face_encoding = get_face_encoding(face_image)
                            face_landmarks_array = np.array(face_landmarks.landmark)

                            if face_encoding is not None:
                                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                
                                if matches[best_match_index]:
                                    name = known_names[best_match_index]
                                    status_text = f"Verified: {name}"
                                    box_color = (0, 255, 0)
                                else:
                                    status_text = "Unknown"
                                    box_color = (0, 0, 255)

                                print(f"Face authenticated: {name}, Depth: {avg_depth:.2f} meters")
                            else:
                                # Fallback to landmarks if encoding is not available
                                min_distance = float('inf')
                                matched_name = "Unknown"
                                
                                for i, known_landmark in enumerate(known_landmarks):
                                    if face_landmarks_array.shape == known_landmark.shape:
                                        dist = calculate_landmark_distance(face_landmarks_array, known_landmark)
                                        if dist < min_distance:
                                            min_distance = dist
                                            matched_name = known_landmark_names[i]

                                threshold = 1  # Adjust this threshold as needed
                                if min_distance < threshold:
                                    status_text = f"Verified: {matched_name}"
                                    box_color = (0, 255, 0)
                                else:
                                    status_text = "Unknown"
                                    box_color = (0, 0, 255)

                                print(f"Face verified by landmarks: {matched_name}, Depth: {avg_depth:.2f} meters")
                                
                            # Draw bounding box and status text
                            cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), box_color, 2)
                            cv2.putText(color_image, status_text, (x_min, y_min-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                            cv2.putText(color_image, f"Depth: {avg_depth:.2f} meters", (x_min, y_max+30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                        else:
                            # If depth is out of range, mark it as not authenticated
                            status_text = "Not Authenticated"
                            box_color = (0, 255, 255)
                            cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), box_color, 2)
                            cv2.putText(color_image, status_text, (x_min, y_min-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                            cv2.putText(color_image, f"Depth: {avg_depth:.2f} meters", (x_min, y_max+30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                            print(f"Face detected but not within depth range.")
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Ensure the pipeline is properly stopped when done
        pipeline.stop()
        cv2.destroyAllWindows()

verify_face()
