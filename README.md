# Face Recognition and Nationality Prediction Application

This Python application uses face recognition to predict the nationality of a person from an image.

## Requirements
- Flask
- Flask-CORS
- Torch
- Torchvision
- Pillow
- opencv-python

## Structure
```
utils/
  preprocess.py       # Contains the preprocessing function for images
model_epoch_41.pth  # Pre-trained PyTorch model
app.py              # Main application file for running the server
requirements.txt    # List of required packages
```
## Installation

1. **Clone the repository and navigate into it:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Download the model:**
    - Download the pre-trained model file `model_epoch_41.pth` from [Google Drive](<https://drive.google.com/file/d/1KyHzFDwMNOpZXAaondadHu9HC9wjh2JX/view?usp=sharing>).
    - Place the downloaded model file in the root directory of the repository.

6. **Run the application:**
    ```bash
    python app.py
    ```

    This will start the Flask server at `http://localhost:5000`.

## Frontend Usage

To test the full functionality of the application, you need to use the frontend provided in a separate GitHub repository. Follow these steps:

1. **Clone the frontend repository:**
    - The frontend code is available [here](<https://github.com/zgeblbl/Nationality-recognition-frontend.git>).
    ```bash
    git clone <frontend_repository_url>
    cd <frontend_directory>
    ```

2. **Set up and start the frontend:**
    - Navigate to the folder containing your frontend files.
    - Double-click the HTML file to open it in your web browser.

3. **Ensure the backend (this application) is running:**
    - The backend should be running on `http://localhost:5000` as described in the previous steps.
    - The frontend is configured to send requests to this address for image processing.

## Testing

1. **Open the frontend application in your web browser:**
    - Double-click the HTML file in your frontend directory to open it in your web browser.

2. **Use the interface to capture an image for nationality prediction:**
    - The frontend will communicate with the backend running on `http://localhost:5000` to process the image and display the prediction results.

## Authors
- Özge Bülbül
- Samet Emin Özen
