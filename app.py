from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import tensorflow as tf
import json
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load single model
model = tf.keras.models.load_model("model/mobilenetv32.keras")
IMG_SIZE = model.input_shape[1]

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_list = [label_map[str(i)] for i in range(len(label_map))]

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Prediction function
def predict(image):
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    input_array = tf.keras.applications.mobilenet_v3.preprocess_input(resized.astype(np.float32))
    input_array = np.expand_dims(input_array, axis=0)
    preds = model.predict(input_array, verbose=0)
    label = label_list[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100
    return label, confidence

# Preprocess and predict function

def preprocess_and_predict(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = image_bgr.shape
        all_x, all_y = [], []

        for lm in results.multi_hand_landmarks[0].landmark:
            all_x.append(int(lm.x * w))
            all_y.append(int(lm.y * h))

        x_min = max(min(all_x) - 20, 0)
        x_max = min(max(all_x) + 20, w)
        y_min = max(min(all_y) - 20, 0)
        y_max = min(max(all_y) + 20, h)

        hand_crop = img_rgb[y_min:y_max, x_min:x_max]

        if hand_crop.size == 0:
            return None, None

        return predict(hand_crop)

    return None, None

# Handle frame socket
@socketio.on("process_frame")
def handle_frame(base64_data):
    try:
        header, encoded = base64_data.split(",", 1)
        img_data = base64.b64decode(encoded)
        img_arr = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        label, confidence = preprocess_and_predict(image)

        if label:
            emit("prediction", {"label": label, "confidence": confidence})
        else:
            emit("prediction", {"label": None, "confidence": 0})
    except Exception as e:
        print("Error processing frame:", e)
        emit("prediction", {"label": None, "confidence": 0})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
