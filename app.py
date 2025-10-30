import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

class DAGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3):
        super(DAGModel, self).__init__()
        self.pathway1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.pathway2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.pathway3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        self.merge = nn.Sequential(
            nn.Linear((hidden_dim // 2) * 2 + (hidden_dim // 4), hidden_dim),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        p1 = self.pathway1(x)
        p2 = self.pathway2(x)
        p3 = self.pathway3(x)
        merged = torch.cat([p1, p2, p3], dim=1)
        return self.merge(merged)

app = Flask(__name__)
CORS(app)
model, scaler, label_encoder, feature_names, class_names = [None] * 5

def load_model():
    global model, scaler, label_encoder, feature_names, class_names
    try:
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        model = DAGModel(input_dim=config['input_dim'], num_classes=config['num_classes'])
        model.load_state_dict(torch.load('production_DAG_Model.pth', map_location='cpu'))
        model.eval()
        with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: label_encoder = pickle.load(f)
        feature_names, class_names = config['feature_names'], config['class_names']
        print("✅ DAG Model loaded successfully!")
    except Exception as e: raise e

@app.route('/')
def home(): return jsonify({"message": "Medication Adherence API", "model": "DAG Model", "status": "active"})

@app.route('/health', methods=['GET'])
def health_check(): return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/info', methods=['GET'])
def model_info(): return jsonify({
    "model_name": "DAG Model", "input_dimensions": len(feature_names),
    "num_classes": len(class_names), "classes": class_names
})

@app.route('/features', methods=['GET'])
def get_features(): return jsonify({"feature_names": feature_names, "num_features": len(feature_names)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data: return jsonify({"error": "No features"}), 400
        patient_features = np.array(data['features'])
        if patient_features.shape[1] != len(feature_names): return jsonify({
            "error": f"Expected {len(feature_names)} features, got {patient_features.shape[1]}"
        }), 400
        features_scaled = scaler.transform(patient_features)
        features_tensor = torch.FloatTensor(features_scaled)
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        predictions = label_encoder.inverse_transform(predicted.numpy())
        return jsonify({
            "predictions": predictions.tolist(),
            "probabilities": probabilities.numpy().tolist(),
            "class_mapping": dict(zip(range(len(class_names)), class_names)),
            "model_used": "DAG Model"
        })
    except Exception as e: return jsonify({"error": str(e)}), 500

print("🚀 Starting API...")
load_model()
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)