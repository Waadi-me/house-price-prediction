# HouseIQ — House Price Prediction

> End-to-end machine learning project predicting residential home sale prices in King County, WA using Linear & Polynomial Regression — complete with a live web interface powered by a Flask backend.

---

## Preview

| Results Dashboard | Live Predictor |
|---|---|
| Tabbed model comparison with figures | Real-time price estimation form |

---

## Features

- **Full ML Pipeline** — data loading, EDA, feature engineering, model training, and evaluation
- **3 Models Compared** — Linear Regression, Polynomial Degree 2, Polynomial Degree 3
- **17 Features** — including engineered features like house age, renovation status, and bath-to-bed ratio
- **Live Predictor** — enter any house's details and get an instant price estimate from the trained model
- **Showcase Website** — professional dark-themed site with tabbed results, animated charts, figure lightbox, and model metrics
- **Output Logging** — all terminal output saved to `output_log.txt` with section headings
- **Figure Export** — all plots saved automatically to `figures/`

---

## 🗂️ Project Structure

```
house-price-prediction/
│
├── index.html                        # Showcase website
├── app.py                            # Flask backend API
├── house_price_prediction.py         # ML training script
├── requirements.txt                  # Python dependencies
├── README.md                         
│
├── model.pkl                         # Trained Linear Regression model
├── imputer.pkl                       # Fitted SimpleImputer
├── label_encoder.pkl                 # Fitted LabelEncoder for cities
│
└── figures/
    ├── eda_plots.png
    ├── correlation_heatmap.png
    ├── linear_regression_results.png
    ├── poly_deg2_results.png
    └── poly_deg3_results.png
```

---

## Dataset

**King County, WA House Sales**
(https://www.kaggle.com/datasets/shree1992/housedata)

| Property | Value |
|---|---|
| Rows | 4,600 |
| Raw Features | 18 |
| Engineered Features | +6 |
| Target Variable | `price` |
| Train / Test Split | 80% / 20% |

**Raw columns:** `date, price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, street, city, statezip, country`

**Engineered features:** `house_age`, `was_renovated`, `yrs_since_renov`, `total_sqft`, `bath_per_bed`, `city_enc`

---

## Model Results

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Linear Regression | $187,400 | $128,200 | 0.6842 |
| Polynomial · Degree 2 | $174,100 | $118,600 | 0.7134 |
| Polynomial · Degree 3 | $171,800 | $115,300 | **0.7209** |

> **Best model:** Polynomial Regression (Degree 3) with Ridge regularization (α=100)

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/waadi-me/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place `data.csv` in the root folder

### 4. Train the model & generate figures
```bash
python house_price_prediction.py
```
This will:
- Train all three models
- Save figures to `figures/`
- Export `model.pkl`, `imputer.pkl`, `label_encoder.pkl`
- Save all output to `output_log.txt`

### 5. Start the Flask server
```bash
python app.py
```

### 6. Open the website
Navigate to `http://127.0.0.1:5000` in your browser.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| ML | scikit-learn, numpy, pandas |
| Visualization | matplotlib, seaborn |
| Backend | Flask, flask-cors |
| Model Persistence | joblib |
| Frontend | HTML, CSS, Vanilla JS |

---

## API Endpoint

**POST** `/predict`

```json
{
  "bedrooms": 3,
  "bathrooms": 2.0,
  "sqft_living": 1800,
  "sqft_lot": 6000,
  "floors": 1.5,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "house_age": 25,
  "was_renovated": 0,
  "yrs_since_renov": 25,
  "city_enc": 17,
  "sale_month": 5
}
```

**Response:**
```json
{
  "success": true,
  "price": 487200,
  "low": 313100,
  "high": 661300,
  "total_sqft": 7800,
  "model": "Linear Regression (scikit-learn)"
}
```

---

## Future Improvements

- [ ] Try XGBoost / Random Forest for higher R²
- [ ] Log-transform the target variable (`price`) for better linear model performance
- [ ] Add geographic price map visualization
- [ ] Deploy to Railway / Render for public access
- [ ] Add model explainability with SHAP values

---

---

## Author

[Wahdat Muhammad Askari](https://github.com/waadi-me)

---

## License

This project is open source and available under the [MIT License](LICENSE).
