from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import io
import base64

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
    if 'images' not in request.files:
        return jsonify({'error': 'No image provided'})

    files = request.files.getlist('images')

    images_base64 = []
    images_html = []

    html_template = """
        <html>
        <head>
            <title>Image Detection Results</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    text-align: center;
                }
                .image-container {
                    margin: 10px;
                    display: inline-block;
                }
                img {
                    max-width: 300px; /* Adjust as needed */
                    max-height: 300px; /* Adjust as needed */
                    width: auto;
                    height: auto;
                    border: 1px solid #ddd; /* Light grey border */
                    border-radius: 4px; /* Rounded border */
                    padding: 5px;
                }
                .image-container p {
                    margin-top: 0;
                }
            </style>
        </head>
        <body>
            <h2>Object Detection Results</h2>
            <div>
                %s
            </div>
        </body>
        </html>
        """
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No image provided'})
        
        #if file:
        #filename = secure_filename(file.filename)


        # Read the image file
        #file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)

         # Ensure the file pointer is reset for next read (important if re-reading file)
        file.seek(0)

        # Convert the image to opencv format (for the YOLO model)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        detection_result = detect_objects(image)
        
           # Convert the result to a bytes-like object
        #_, img_encoded = cv2.imencode('.png', detection_result)
        #img_bytes = io.BytesIO(img_encoded).read()

        # Encode the image to base64 string and add to the list
        #img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        #images_base64.append(img_base64)
        #print(images_base64)\
        
        #buffered = io.BytesIO()
        #cv2.imwrite(buffered, detection_result, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        #img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        _, buffer = cv2.imencode('.jpg', detection_result, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        img_str = base64.b64encode(buffer).decode('utf-8')

        
        images_html.append(f'<div class="image-container"><img src="data:image/jpeg;base64,{img_str}"/><p>{file.filename}</p></div>')

    html_content = html_template % ''.join(images_html)
    return render_template_string(html_content)

    # Return the image file
    #return send_file(img_bytes, mimetype='image/png')
    #return jsonify(images_base64)

