#to find camera index

import cv2

def detect_camera_indices(max_indices=10):
    available_indices = []
    for index in range(max_indices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera index {index} is available.")
            available_indices.append(index)
            cap.release()
        else:
            print(f"Camera index {index} is not available.")
    return available_indices

if __name__ == "__main__":
    # Adjust the maximum number of indices to check based on your setup
    available_cameras = detect_camera_indices()
    print("Available camera indices:", available_cameras)
