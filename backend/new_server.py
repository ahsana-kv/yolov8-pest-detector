from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLOv8 model
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
   
    file = request.files['file']
    
    # Read image directly from file stream
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run inference
    results = model(image)

    # Extract predictions
    predictions = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            predictions.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'confidence': conf, 
                'class_id': int(cls),
                'class_name': model.names[int(cls)]  # Add class name
            })

    return jsonify({
        'predictions': predictions,
        'original_size': image.size  # Return original image dimensions
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)