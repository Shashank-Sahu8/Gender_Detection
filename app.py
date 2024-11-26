from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import base64
import io
from PIL import Image
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('gender_detection.keras')
classes = ['man', 'woman']

# Load OpenCV's pre-trained face detection model (Haar Cascade or DNN)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_data = request.json.get("image")
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Decode the image
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img = img.convert("RGB")
        img = np.array(img)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If no faces are detected, return an error
        if len(faces) == 0:
            return jsonify({"error": "No face detected in the image"}), 400

        # Process the first face detected
        (startX, startY, width, height) = faces[0]
        face_crop = img[startY:startY + height, startX:startX + width]

        # Preprocess the face image for gender detection
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict gender
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        confidence = conf[idx] * 100

        # Return the result
        return jsonify({"gender": label, "confidence": float(confidence)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
