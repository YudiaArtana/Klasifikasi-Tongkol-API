import os
import io
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Load model and preprocessing objects
def load_all_objects(filename):
    data = joblib.load(filename)
    print(f"Scalers, komponen PCA, dan model KNN dimuat dari {filename}")
    return data['scalers'], data['pca_components'], data['knn_model']

scalers, pca_components, knn_model = load_all_objects('model.pkl')

# Flask app setup
app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def transform_new_image_with_2dpca_per_channel(image_scaled, pca_components):
    mean_image = pca_components['Mean Image']
    mean_proj_width = pca_components['Mean Projection']
    n_samples, height, width = image_scaled.shape
    image_centered = image_scaled - mean_image
    width_components = pca_components['Width'][0]
    height_components = pca_components['Height'][0]
    width_projection = np.array([image_centered[i] @ width_components for i in range(n_samples)])
    width_projection_centered = width_projection - mean_proj_width
    height_projection = np.array([width_projection_centered[i].T @ height_components for i in range(n_samples)])
    return height_projection

@app.route('/')
def hello_world():
    return jsonify('API IS RUNNING!')

@app.route('/predict', methods=['POST'])
def predict():
    labels = {0: 'SEGAR', 1: 'TIDAK SEGAR'}
    img_file = request.files.get('file')
    
    if img_file and allowed_file(img_file.filename):
        img_bytes = img_file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": {"code": 400, "message": "Gambar tidak valid"}})
        
        img_resized = cv2.resize(img, (128, 128))
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        img_hsv = img_hsv[np.newaxis, ...]
        
        hue_features = img_hsv[:, :, :, 0]
        saturation_features = img_hsv[:, :, :, 1]
        value_features = img_hsv[:, :, :, 2]
        
        hue_scaled = scalers['hue'].transform(hue_features.reshape(1, -1)).reshape(1, 128, 128)
        saturation_scaled = scalers['saturation'].transform(saturation_features.reshape(1, -1)).reshape(1, 128, 128)
        value_scaled = scalers['value'].transform(value_features.reshape(1, -1)).reshape(1, 128, 128)
        
        transformed_hue = transform_new_image_with_2dpca_per_channel(hue_scaled, pca_components['hue'])
        transformed_saturation = transform_new_image_with_2dpca_per_channel(saturation_scaled, pca_components['saturation'])
        transformed_value = transform_new_image_with_2dpca_per_channel(value_scaled, pca_components['value'])
        
        X_color_2dpca_flat = np.concatenate(
            [transformed_hue.reshape(1, -1),
             transformed_saturation.reshape(1, -1),
             transformed_value.reshape(1, -1)],
            axis=-1)
        
        probabilities = knn_model.predict_proba(X_color_2dpca_flat)
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[0][predicted_class] * 100
        
        if confidence < 50:
            return jsonify({"status": {"code": 400, "message": "Gambar kurang jelas / tidak terdeteksi"}})
        
        return jsonify({
            "status": {"code": 200, "message": "Success predicting"},
            "prediction": {"class_name": labels[predicted_class], "confidence": f"{confidence:.2f}%"}
        })
    
    return jsonify({"status": {"code": 400, "message": "No image uploaded"}})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
