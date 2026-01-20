from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")

def find_direction(x_center, frame_w):
    if x_center < frame_w * 0.33:
        return "left"
    elif x_center > frame_w * 0.66:
        return "right"
    else:
        return "center"

@app.route("/")
def home():
    return "Intelligent Navigation Aid API is running"

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    image_data = data["image"].split(",")[1]
    decoded = base64.b64decode(image_data)

    npimg = np.frombuffer(decoded, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            x_center = (x1 + x2) / 2
            direction = find_direction(x_center, frame.shape[1])

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "direction": direction,
                "distance": "approx 2 meters"
            })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
