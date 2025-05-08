from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms as transforms
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load YOLOv8 model
model = YOLO("best.pt")  # Correct way to load a YOLOv8 model

# Define image preprocessing (YOLO expects image arrays, no need for PyTorch transforms)
def preprocess_image(image):
    return image  # No need to apply manual transformations for YOLOv8

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
   
    file = request.files['file']
    image = Image.open(file).convert("RGB")

    # Run inference
    results = model(image)

    # Extract predictions (bounding boxes, class IDs, etc.)
    predictions = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            predictions.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': conf, 'class_id': int(cls)
            })

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
