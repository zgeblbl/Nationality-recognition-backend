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
        images = data['images']  # Expecting a list of base64 image strings
        
        predictions = []
        
        for image_data in images:
            image_data = image_data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                prediction = model(input_tensor)
                predicted_class = torch.argmax(prediction, dim=1).item()
                predictions.append(predicted_class)

        # Mapping for class labels
        mapping = {0: 'Arabic', 1: 'Chinese', 2: 'Czech', 3: 'Dutch', 4: 'English', 
                   5: 'French', 6: 'German', 7: 'Honduran', 8: 'Italian', 9: 'Japanese', 
                   10: 'Mexican', 11: 'Russian', 12: 'Spanish', 13: 'Türk'}

        # Find the most common prediction
        print(predictions)
        most_common_prediction = Counter(predictions).most_common(1)[0][0]
        result = mapping.get(most_common_prediction, 'Unknown')

        return jsonify({'prediction': result})

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Failed to process images or make a prediction.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
