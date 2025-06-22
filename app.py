from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import tensorflow as tf
import json
import io
from PIL import Image
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model & labels
model1 = tf.keras.models.load_model("model/sgd.keras")
model2 = tf.keras.models.load_model("model/sgd2.keras")
model3 = tf.keras.models.load_model("model/sgd3.keras")
IMG_SIZE = model1.input_shape[1]

IMG_SIZE = model1.input_shape[1]
with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_list = [label_map[str(i)] for i in range(len(label_map))]

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

from collections import Counter

def predict_with_voting(image):
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    input_array = tf.keras.applications.mobilenet_v3.preprocess_input(resized.astype(np.float32))
    input_array = np.expand_dims(input_array, axis=0)

    preds1 = model1.predict(input_array, verbose=0)
    preds2 = model2.predict(input_array, verbose=0)
    preds3 = model3.predict(input_array, verbose=0)

    label1 = label_list[np.argmax(preds1)]
    label2 = label_list[np.argmax(preds2)]
    label3 = label_list[np.argmax(preds3)]

    vote_result = Counter([label1, label2, label3]).most_common(1)[0]
    final_label = vote_result[0]
    vote_count = vote_result[1]
    confidence = (vote_count / 3) * 100

    return final_label, confidence


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

        resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
        input_array = tf.keras.applications.mobilenet_v3.preprocess_input(resized.astype(np.float32))
        input_array = np.expand_dims(input_array, axis=0)

        label, confidence = predict_with_voting(hand_crop)
        return label, confidence

    return None, None

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
