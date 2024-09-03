import cv2
import face_recognition
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
import os

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
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"Successfully verified that {filename} exists and is not empty.")
        else:
            print(f"Warning: {filename} does not exist or is empty.")
    except Exception as e:
        print(f"Error saving face encoding: {e}")

def save_landmarks(landmarks, filename):
    """Save the facial landmarks to a file."""
    try:
        np.save(filename, landmarks)
        print(f"Facial landmarks saved to {filename}.")
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            print(f"Successfully verified that {filename} exists and is not empty.")
        else:
            print(f"Warning: {filename} does not exist or is empty.")
    except Exception as e:
        print(f"Error saving facial landmarks: {e}")

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

def calculate_avg_depth(face_landmarks, depth_frame, color_image_shape):
    """Calculate the average depth of the face region."""
    depth_values = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * color_image_shape[1])
        y = int(lm.y * color_image_shape[0])
        depth = depth_frame.get_distance(x, y)
        depth_values.append(depth)
    return np.mean(depth_values)

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

            # Extract face landmarks
            face_landmarks_array = get_face_landmarks(results)
            
            # Calculate the average depth of the face region
            avg_depth = calculate_avg_depth(face_landmarks, depth_frame, color_image.shape)
            print(f"Average depth of face: {avg_depth:.2f} meters")

            # Ensure the detected face is within a reasonable depth range
            if avg_depth == 0:
                print("Depth is zero. Cannot enroll.")
                continue

            if 0 < avg_depth <= 7:
                face_image = color_image
                face_encoding = get_face_encoding(face_image)
                
                if face_encoding is not None:
                    encoding_filename = os.path.join(encoding_folder, f'{name}.npy')
                    save_face_encoding(face_encoding, encoding_filename)
                    
                    landmarks_filename = os.path.join(landmarks_folder, f'{name}_landmarks.npy')
                    save_landmarks(face_landmarks_array, landmarks_filename)
                    
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
