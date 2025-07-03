from flask import Flask
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import cv2
import tensorflow as tf
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
model = tf.keras.models.load_model("model/mobilenetv32.keras")
IMG_SIZE = model.input_shape[1]  # Asumsi model input shape (224, 224, 3)

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
label_list = [label_map[str(i)] for i in range(len(label_map))]

def predict(image):
    """
    Simplified prediction function that expects pre-segmented hand image
    """
    # Resize to model input size
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Convert to RGB if needed (frontend sends black background with red/green landmarks)
    if image.shape[2] == 1:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Preprocess for MobileNetV3
    input_array = tf.keras.applications.mobilenet_v3.preprocess_input(
        resized.astype(np.float32))
    input_array = np.expand_dims(input_array, axis=0)
    
    # Predict
    preds = model.predict(input_array, verbose=0)
    label = label_list[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100
    
    return label, confidence

@socketio.on("process_frame")
def handle_frame(base64_data):
    try:
        # Decode base64 image
        header, encoded = base64_data.split(",", 1)
        img_data = base64.b64decode(encoded)
        img_arr = np.frombuffer(img_data, dtype=np.uint8)
        
        # Decode image (expecting black background with hand landmarks)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Direct prediction (no hand detection needed)
        label, confidence = predict(image_rgb)
        
        if label and confidence > 0:
            emit("prediction", {
                "label": label,
                "confidence": confidence
            })
        else:
            emit("prediction", {
                "label": "unknown",
                "confidence": 0
            })
            
    except Exception as e:
        print("Error processing frame:", str(e))
        emit("prediction", {
            "label": "error",
            "confidence": 0
        })

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)