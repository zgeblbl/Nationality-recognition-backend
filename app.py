from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import Counter
import logging
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained ResNet model
model = models.resnet18(weights=None)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=13)

# Load the state dict
try:
    checkpoint = torch.load('model_epoch_41.pth', map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    model.load_state_dict(state_dict, strict=False)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

model.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        images = data.get('images', [])  # Expecting a list of base64 image strings
        
        if not images:
            return jsonify({'prediction': 'No face detected'})

        predictions = []
        no_face_count = 0

        for image_data in images:
            image_data = image_data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            open_cv_image = np.array(image) 
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # If no faces are detected, skip this image
            if len(faces) == 0:
                no_face_count += 1
                continue  # Skip processing this image

            # Preprocess the image
            input_tensor = transform(image).unsqueeze(0)

            # Run the model for prediction
            with torch.no_grad():
                prediction = model(input_tensor)
                predicted_class = torch.argmax(prediction, dim=1).item()
                predictions.append(predicted_class)

        # Class label mapping
        mapping = {0: 'Arabic', 1: 'Chinese', 2: 'Czech', 3: 'Dutch', 4: 'English', 
                   5: 'French', 6: 'German', 7: 'Honduran', 8: 'Italian', 9: 'Japanese', 
                   10: 'Mexican', 11: 'Russian', 12: 'Spanish', 13: 'Türk'}

        # If no valid faces were detected in all images
        if no_face_count == len(images):
            return jsonify({'prediction': 'No face detected in any of the images'})

        # If predictions were made
        if len(predictions) == 0:
            return jsonify({'prediction': 'Try Again'})

        # Find the most common prediction
        most_common_prediction = Counter(predictions).most_common(1)[0][0]
        result = mapping.get(most_common_prediction, 'Unknown')

        # Return all predictions and the most common one
        return jsonify({
            'predictions': [mapping.get(pred, 'Unknown') for pred in predictions],
            'most_common_prediction': result
        })

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': f'Failed to process images or make a prediction: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
