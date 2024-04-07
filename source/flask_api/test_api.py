import os
import requests

# Specify the URL of your Flask API
api_url = 'http://127.0.0.1:5000/detect'  # Update with your actual API URL

# Get the absolute path to the image file
image_path = os.path.abspath('/flask_api/img1.jpg')  # Update with the actual image file name

# Send a POST request with the image file
files = {'file': open(image_path, 'rb')}
response = requests.post(api_url, files=files)

# Check the response
print(response.json())
