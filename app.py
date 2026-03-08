from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os
from flask import send_from_directory

app = Flask(__name__, static_folder='.')
CORS(app)

# ── Load model artifacts ───────────────────────────────────────────────────────
model         = joblib.load('model.pkl')
imputer       = joblib.load('imputer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ── Serve index.html at root ───────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/figures/<path:filename>')
def figures(filename):
    return send_from_directory('figures', filename)

# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        bedrooms        = float(data['bedrooms'])
        bathrooms       = float(data['bathrooms'])
        sqft_living     = float(data['sqft_living'])
        sqft_lot        = float(data['sqft_lot'])
        floors          = float(data['floors'])
        waterfront      = float(data['waterfront'])
        view            = float(data['view'])
        condition       = float(data['condition'])
        sqft_above      = float(data['sqft_above'])
        sqft_basement   = float(data['sqft_basement'])
        house_age       = float(data['house_age'])
        was_renovated   = float(data['was_renovated'])
        yrs_since_renov = float(data['yrs_since_renov'] if data['was_renovated'] else house_age)
        city_enc        = float(data['city_enc'])
        sale_month      = float(data['sale_month'])

        # Derived features (same as notebook)
        total_sqft  = sqft_living + sqft_lot
        bath_per_bed = bathrooms / max(bedrooms, 1)

        features = np.array([[
            bedrooms, bathrooms, sqft_living, sqft_lot,
            floors, waterfront, view, condition,
            sqft_above, sqft_basement,
            house_age, was_renovated, yrs_since_renov,
            total_sqft, bath_per_bed, city_enc, sale_month
        ]])

        # Impute then predict
        features_imputed = imputer.transform(features)
        predicted_price  = model.predict(features_imputed)[0]
        rmse             = 174100  # from your notebook results

        return jsonify({
            'success':    True,
            'price':      round(float(predicted_price)),
            'low':        round(float(max(predicted_price - rmse, 50000))),
            'high':       round(float(predicted_price + rmse)),
            'total_sqft': int(total_sqft),
            'model':      'Linear Regression (scikit-learn)'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
