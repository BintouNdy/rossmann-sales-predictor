from flask import Flask, request, jsonify
import pandas as pd
from src.features import FEATURES, load_model
from src.predict import predict_sales

# Initialisation
app = Flask(__name__)

# Chargement du modèle
model = load_model()


@app.route('/')
def home():
    return "✅ API Rossmann prête. Utilisez POST /predict pour envoyer vos données."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({'error': 'Aucune donnée reçue'}), 400

        prediction = predict_sales(input_data)
        if isinstance(prediction, float):
            return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': prediction}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)