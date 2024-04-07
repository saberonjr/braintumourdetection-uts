from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import io

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolo_50eoch.pt')

def detect_objects(image):
    # Run inference on the input image
    results = model(image)

    # Process each result in the list
    for result in results:
        # Plot the detection results on the original image
        annotated_image = result.plot()

        # Convert the annotated image to bytes
        _, img_encoded = cv2.imencode('.png', annotated_image)

    return img_encoded

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    # Read the image file
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)

    # Convert the image to opencv format (for the YOLO model)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    detection_results = detect_objects(image)

    # Convert the bytes to a file-like object
    img_bytes = io.BytesIO(detection_results)

    # Return the image file
    return send_file(img_bytes, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='192.168.254.101')