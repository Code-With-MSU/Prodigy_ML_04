import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Setup Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model_path = 'gesture_recognizer.task'
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# MediaPipe hand connections
mp_hands = mp.solutions.hands
hand_connections = mp_hands.HAND_CONNECTIONS

# Recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process image
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save uploaded file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
        file.save(filepath)

        # Read image
        bgr_image = cv2.imread(filepath)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Recognize gestures
        with GestureRecognizer.create_from_options(options) as recognizer:
            result = recognizer.recognize(mp_image)

            # Draw landmarks and connections
            if result.hand_landmarks:
                h, w, _ = bgr_image.shape
                for hand_landmarks in result.hand_landmarks:
                    for landmark in hand_landmarks:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(bgr_image, (cx, cy), 5, (0, 255, 0), -1)

                    for connection in hand_connections:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]
                        x1, y1 = int(start.x * w), int(start.y * h)
                        x2, y2 = int(end.x * w), int(end.y * h)
                        cv2.line(bgr_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Display gesture name
            gesture_text = "Gesture: Not Detected"
            if result.gestures and result.gestures[0]:
                gesture_name = result.gestures[0][0].category_name
                gesture_text = f'Gesture: {gesture_name}'

        # Resize image bigger (1.5x)
        bgr_image = cv2.resize(bgr_image, (0, 0), fx=1.5, fy=1.5)

        # Save the processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        cv2.imwrite(output_path, bgr_image)

        # Return the output image and gesture name on the web page
        return render_template('index.html', filename='output.jpg', gesture=gesture_text)

if __name__ == "__main__":
    app.run(debug=True)
