from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# Load the pre-trained ResNet model
model = models.resnet18(pretrained=False)

# Redefine the last layer to match the number of output classes (5 in this case)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=5)

# Load the state dict, ignoring the size mismatch for the final layer
checkpoint = torch.load('model_epoch_41.pth', map_location=torch.device('cpu'))

# Use strict=False to load only compatible layers
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Set the model to evaluation mode
model.eval()

# Preprocessing function (adjust according to your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Decode the base64 image
    image_data = data['image'].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Preprocess the image and prepare for model input
    input_tensor = transform(image).unsqueeze(0)

    # Get the prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Return the prediction as JSON
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
