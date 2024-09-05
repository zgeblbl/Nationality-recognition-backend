from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained ResNet model
model = models.resnet18(weights=None)  # Use weights=None as 'pretrained' is deprecated
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=13)

# Load the state dict
checkpoint = torch.load('model_epoch_41.pth', map_location=torch.device('cpu'))

# Remove the final layer from the state dictionary
print(checkpoint.keys())
state_dict = checkpoint['model_state_dict']
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)

# Load state dict into the model
try:
    model.load_state_dict(state_dict, strict=False)
except Exception as e:
    print("Error loading state_dict:", str(e))

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
        image_data = data['image'].split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()

            # make mapping {'Arabic': 0, 'Chinese': 1, 'Czech': 2, 'Dutch': 3, 'English': 4, 'French': 5, 'German': 6, 'Honduran': 7, 'Italian': 8, 'Japanese': 9, 'Mexican': 10, 'Russian': 11, 'Spanish': 12, 'Türk': 13}
            print(predicted_class)
            mapping = {0: 'Arabic', 1: 'Chinese', 2: 'Czech', 3: 'Dutch', 4: 'English', 5: 'French', 6: 'German', 7: 'Honduran', 8: 'Italian', 9: 'Japanese', 10: 'Mexican', 11: 'Russian', 12: 'Spanish', 13: 'Türk'}

        return jsonify({'prediction': mapping[predicted_class]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
