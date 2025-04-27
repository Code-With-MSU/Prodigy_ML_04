import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Load model
model_path = 'gesture_recognizer.task'

# Import classes
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe hand connections
mp_hands = mp.solutions.hands
hand_connections = mp_hands.HAND_CONNECTIONS

# Create recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

# Read the image
image_path = 'thumbs up.jpeg'  # <-- Correct path to your image
bgr_image = cv2.imread(image_path)

# Check if image is loaded
if bgr_image is None:
    print("Error: Image not found.")
    exit()

# Convert BGR to RGB
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# Create MediaPipe Image
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# Create Gesture Recognizer
with GestureRecognizer.create_from_options(options) as recognizer:
    # Recognize gestures
    result = recognizer.recognize(mp_image)

    # Draw landmarks and connections
    if result.hand_landmarks:
        h, w, _ = bgr_image.shape
        for hand_landmarks in result.hand_landmarks:
            # Draw circles
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(bgr_image, (cx, cy), 5, (0, 255, 0), -1)

            # Draw connections
            for connection in hand_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(bgr_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display gesture name if detected
    gesture_text = "Not Detected"
    if result.gestures and result.gestures[0]:
        gesture_name = result.gestures[0][0].category_name
        gesture_score = result.gestures[0][0].score
        gesture_text = f' {gesture_name}'

  

    # Put text on image
    cv2.putText(bgr_image, gesture_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 🔥 Resize the image bigger
    bgr_image = cv2.resize(bgr_image, None, fx=1.5, fy=1.5)

    # Show final image
    cv2.imshow('Hand Gesture Recognition', bgr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
