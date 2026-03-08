#IMPORTING LIBRARIES AND OTHER STUFF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
print('✅ Libraries loaded!')

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f'✅ Figures will be saved to: {os.path.abspath(FIGURES_DIR)}')

LOG_FILE = 'output_log.txt'
log_f = open(LOG_FILE, 'w', encoding='utf-8')

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, log_f)
print('=' * 60)
print('   HOUSE PRICE PREDICTION — OUTPUT LOG')
print('=' * 60)

#IMPORTING DATASET
df = pd.read_csv('data.csv')

print(f'✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns')
df.head(3)

print('\n' + '=' * 60)
print('  SECTION 3: EXPLORATORY DATA ANALYSIS')
print('=' * 60)
print('=== Data Types ===')
print(df.dtypes)
print('\n=== Missing Values ===')
print(df.isnull().sum())
print('\n=== Price Statistics ===')
print(df['price'].describe())

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis — House Prices', fontsize=16, fontweight='bold')

# 1. Price Distribution
axes[0,0].hist(df['price'], bins=60, color='steelblue', edgecolor='white')
axes[0,0].set_title('Price Distribution')
axes[0,0].set_xlabel('Price ($)')
axes[0,0].set_ylabel('Count')

# 2. Log Price Distribution
axes[0,1].hist(np.log1p(df['price']), bins=60, color='coral', edgecolor='white')
axes[0,1].set_title('Log(Price) Distribution')
axes[0,1].set_xlabel('log(Price)')

# 3. sqft_living vs price
axes[0,2].scatter(df['sqft_living'], df['price'], alpha=0.2, color='mediumseagreen', s=10)
axes[0,2].set_title('Sqft Living vs Price')
axes[0,2].set_xlabel('Sqft Living')
axes[0,2].set_ylabel('Price ($)')

# 4. Bedrooms vs Price
df.boxplot(column='price', by='bedrooms', ax=axes[1,0])
axes[1,0].set_title('Bedrooms vs Price')
axes[1,0].set_xlabel('Bedrooms')
axes[1,0].set_ylabel('Price ($)')
plt.sca(axes[1,0]); plt.title('Bedrooms vs Price')

# 5. Condition vs Price
df.boxplot(column='price', by='condition', ax=axes[1,1])
axes[1,1].set_title('Condition vs Price')
axes[1,1].set_xlabel('Condition (1-5)')
plt.sca(axes[1,1]); plt.title('Condition vs Price')

# 6. Waterfront vs Price
df.boxplot(column='price', by='waterfront', ax=axes[1,2])
axes[1,2].set_title('Waterfront vs Price')
axes[1,2].set_xlabel('Waterfront (0=No, 1=Yes)')
plt.sca(axes[1,2]); plt.title('Waterfront vs Price')

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlation heatmap
numeric_cols = ['price','bedrooms','bathrooms','sqft_living','sqft_lot',
                'floors','waterfront','view','condition',
                'sqft_above','sqft_basement','yr_built','yr_renovated']

plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print('\n📌 Top features correlated with price:')
print(corr['price'].sort_values(ascending=False).drop('price'))

df2 = df.copy()

# ── Date features ─────────────────────────────────────────────────────────────
df2['date'] = pd.to_datetime(df2['date'])
df2['sale_year']  = df2['date'].dt.year
df2['sale_month'] = df2['date'].dt.month

# ── Derived features ──────────────────────────────────────────────────────────
df2['house_age']       = df2['sale_year'] - df2['yr_built']
df2['was_renovated']   = (df2['yr_renovated'] > 0).astype(int)
df2['yrs_since_renov'] = np.where(
    df2['yr_renovated'] > 0,
    df2['sale_year'] - df2['yr_renovated'],
    df2['house_age']
)
df2['total_sqft']   = df2['sqft_living'] + df2['sqft_lot']
df2['bath_per_bed'] = df2['bathrooms'] / (df2['bedrooms'].replace(0, 1))

# ── Encode city (top 20 cities, rest = 'Other') ───────────────────────────────
top_cities = df2['city'].value_counts().nlargest(20).index
df2['city_enc'] = df2['city'].apply(lambda x: x if x in top_cities else 'Other')
le = LabelEncoder()
df2['city_enc'] = le.fit_transform(df2['city_enc'])

print('\n' + '=' * 60)
print('  SECTION 4: FEATURE ENGINEERING')
print('=' * 60)
print('✅ Feature engineering complete!')
print('New features: house_age, was_renovated, yrs_since_renov, total_sqft, bath_per_bed, city_enc')

FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition',
    'sqft_above', 'sqft_basement',
    'house_age', 'was_renovated', 'yrs_since_renov',
    'total_sqft', 'bath_per_bed', 'city_enc', 'sale_month'
]
TARGET = 'price'

X = df2[FEATURES]
y = df2[TARGET]

imputer = SimpleImputer(strategy='median')
X_clean = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

print(f'✅ Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples')
print(f'✅ Features used: {len(FEATURES)}')

print('\n' + '=' * 60)
print('  SECTION 5: LINEAR REGRESSION')
print('=' * 60)
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f'\n📊 {name}')
    print(f'   RMSE : ${rmse:,.0f}  (avg prediction error)')
    print(f'   MAE  : ${mae:,.0f}  (avg absolute error)')
    print(f'   R²   : {r2:.4f}      (1.0 = perfect)')
    return y_pred, r2

lr = LinearRegression()
y_pred_lr, r2_lr = evaluate_model('Linear Regression', lr, X_train, X_test, y_train, y_test)

# Feature importance
coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': lr.coef_})
coef_df = coef_df.iloc[coef_df['Coefficient'].abs().argsort()[::-1]]
print('\n📌 Top Feature Coefficients:')
print(coef_df.head(10).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Linear Regression Results', fontsize=14, fontweight='bold')

# Actual vs Predicted
axes[0].scatter(y_test, y_pred_lr, alpha=0.3, color='steelblue', s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price ($)')
axes[0].set_ylabel('Predicted Price ($)')
axes[0].set_title(f'Actual vs Predicted  (R²={r2_lr:.3f})')

# Residuals
residuals = y_test.values - y_pred_lr
axes[1].scatter(y_pred_lr, residuals, alpha=0.3, color='coral', s=10)
axes[1].axhline(0, color='black', lw=1.5, linestyle='--')
axes[1].set_xlabel('Predicted Price ($)')
axes[1].set_ylabel('Residuals ($)')
axes[1].set_title('Residual Plot')

# Feature importance
top10 = coef_df.head(10)
colors = ['steelblue' if c > 0 else 'coral' for c in top10['Coefficient']]
axes[2].barh(top10['Feature'], top10['Coefficient'], color=colors)
axes[2].set_title('Top 10 Feature Coefficients')
axes[2].set_xlabel('Coefficient Value')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/linear_regression_results.png', dpi=150, bbox_inches='tight')
plt.show()

results = {'Linear Regression': r2_lr}

print('\n' + '=' * 60)
print('  SECTION 6: POLYNOMIAL REGRESSION')
print('=' * 60)
for degree in [2, 3]:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
        ('model',  Ridge(alpha=100))
    ])
    y_pred_poly, r2 = evaluate_model(
        f'Polynomial Regression (degree={degree})',
        pipe, X_train, X_test, y_train, y_test
    )
    results[f'Polynomial (deg {degree})'] = r2

    # Save figure for this degree
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Polynomial Regression (Degree {degree}) Results', fontsize=14, fontweight='bold')

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred_poly, alpha=0.3, color='steelblue', s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Price ($)')
    axes[0].set_ylabel('Predicted Price ($)')
    axes[0].set_title(f'Actual vs Predicted  (R²={r2:.3f})')

    # Residuals
    residuals_poly = y_test.values - y_pred_poly
    axes[1].scatter(y_pred_poly, residuals_poly, alpha=0.3, color='coral', s=10)
    axes[1].axhline(0, color='black', lw=1.5, linestyle='--')
    axes[1].set_xlabel('Predicted Price ($)')
    axes[1].set_ylabel('Residuals ($)')
    axes[1].set_title('Residual Plot')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/poly_deg{degree}_results.png', dpi=150, bbox_inches='tight')
    plt.show()

plt.figure(figsize=(8, 4))
colors = ['steelblue', 'coral', 'mediumseagreen']
bars = plt.barh(list(results.keys()), list(results.values()), color=colors)
plt.xlabel('R² Score')
plt.title('Model Comparison — R² Score (higher = better)')
plt.xlim(0, 1.05)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

best = max(results, key=results.get)
print('\n' + '=' * 60)
print('  SECTION 7: MODEL COMPARISON')
print('=' * 60)
print(f'\n🏆 Best Model: {best}  |  R² = {results[best]:.4f}')

# Customize these values for any house you want to predict
# ── User input for new house prediction ───────────────────────────────────────
print('\n' + '=' * 60)
print('  SECTION 8: PREDICT A NEW HOUSE')
print('=' * 60)
print('Enter the details of the house you want to predict.')
print('(Valid ranges are shown in brackets)\n')

def get_input(prompt, min_val, max_val, dtype=float):
    while True:
        try:
            val = dtype(input(f'  {prompt} [{min_val}–{max_val}]: '))
            if min_val <= val <= max_val:
                return val
            print(f'    ⚠️  Please enter a value between {min_val} and {max_val}.')
        except ValueError:
            print(f'    ⚠️  Invalid input. Please enter a number.')

bedrooms        = get_input('Bedrooms',                   1,    10,   int)
bathrooms       = get_input('Bathrooms',                  0.5,  6.5,  float)
sqft_living     = get_input('Sqft Living (sq ft)',        200,  10000,int)
sqft_lot        = get_input('Sqft Lot (sq ft)',           500,  150000,int)
floors          = get_input('Floors',                     1,    3.5,  float)
waterfront      = get_input('Waterfront (0=No, 1=Yes)',   0,    1,    int)
view            = get_input('View quality',               0,    4,    int)
condition       = get_input('Condition',                  1,    5,    int)
sqft_above      = get_input('Sqft Above ground (sq ft)',  200,  10000,int)
sqft_basement   = get_input('Sqft Basement (sq ft)',      0,    5000, int)
house_age       = get_input('House Age (years)',          0,    125,  int)
was_renovated   = get_input('Was Renovated (0=No, 1=Yes)',0,    1,    int)
yrs_since_renov = get_input('Years Since Renovation',     0,    125,  int)
total_sqft      = sqft_living + sqft_lot   # auto-calculated
bath_per_bed    = round(bathrooms / max(bedrooms, 1), 2)  # auto-calculated
sale_month      = get_input('Sale Month',                 1,    12,   int)

# City selection
city_names = le.classes_
print(f'\n  Available cities:')
for i, name in enumerate(city_names):
    print(f'    {i:2d} = {name}')
city_enc = get_input(f'City (enter number)', 0, len(city_names) - 1, int)


new_house = pd.DataFrame([{
    'bedrooms':        bedrooms,
    'bathrooms':       bathrooms,
    'sqft_living':     sqft_living,
    'sqft_lot':        sqft_lot,
    'floors':          floors,
    'waterfront':      waterfront,
    'view':            view,
    'condition':       condition,
    'sqft_above':      sqft_above,
    'sqft_basement':   sqft_basement,
    'house_age':       house_age,
    'was_renovated':   was_renovated,
    'yrs_since_renov': yrs_since_renov,
    'total_sqft':      total_sqft,
    'bath_per_bed':    bath_per_bed,
    'city_enc':        city_enc,
    'sale_month':      sale_month,
}])

print('\n' + '=' * 60)
print('  SECTION 8: USER INPUT SUMMARY')
print('=' * 60)
print(f'  Bedrooms              : {bedrooms}')
print(f'  Bathrooms             : {bathrooms}')
print(f'  Sqft Living           : {sqft_living} sq ft')
print(f'  Sqft Lot              : {sqft_lot} sq ft')
print(f'  Floors                : {floors}')
print(f'  Waterfront            : {"Yes" if waterfront else "No"}')
print(f'  View Quality          : {view}/4')
print(f'  Condition             : {condition}/5')
print(f'  Sqft Above Ground     : {sqft_above} sq ft')
print(f'  Sqft Basement         : {sqft_basement} sq ft')
print(f'  House Age             : {house_age} years')
print(f'  Was Renovated         : {"Yes" if was_renovated else "No"}')
print(f'  Years Since Renovation: {yrs_since_renov} years')
print(f'  Total Sqft (auto)     : {total_sqft} sq ft')
print(f'  Bath per Bedroom (auto): {bath_per_bed}')
print(f'  City                  : {city_names[city_enc]}')
print(f'  Sale Month            : {sale_month}')

predicted = lr.predict(new_house)[0]
print(f'\n  🏠 Predicted Sale Price: ${predicted:,.0f}')

# ── Close log file ─────────────────────────────────────────────────────────
sys.stdout = sys.stdout.streams[0]  # restore original stdout
log_f.close()
print(f'✅ Output log saved to: {LOG_FILE}')

joblib.dump(lr, 'model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(le, 'label_encoder.pkl')
print('✅ Model saved to model.pkl')