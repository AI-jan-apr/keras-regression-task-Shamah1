# House Price Prediction Task

A regression model using Keras to predict house prices in King County, Washington.

---

## Project Structure
```
01-Keras-Regression-task.ipynb
deploy.py
model_weights.pkl
scaler_weights.pkl
```

---

## Model

- **Dataset**: KC House Data (21,613 samples, 20 features)
- **Architecture**: Dense(19) × 4 → Dense(1)
- **Optimizer**: Adam | **Loss**: MSE
- **MAE**: ~$101,655 | **Explained Variance**: 81%

---

## Run the API
```
uvicorn deploy:app --reload
```

---

## Endpoints

### GET `/`
Returns a welcome message and available endpoints.

### POST `/predict`
Send house features and get the predicted price with a confidence range.

**Example Input:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2.25,
  "sqft_living": 2570,
  "sqft_lot": 7242,
  "floors": 2.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_above": 2170,
  "sqft_basement": 400,
  "yr_built": 1951,
  "yr_renovated": 1991,
  "zipcode": 98125,
  "lat": 47.721,
  "long": -122.319,
  "sqft_living15": 1690,
  "sqft_lot15": 7639,
  "month": 12,
  "year": 2014
}
```

**Example Output:**
```json
{
  "predicted_price": "$608,755",
  "range": "Between $507,101 and $710,410",
  "note": "Based on model MAE of $101,655"
}
```

### GET `/evaluate`
Returns model performance metrics.

**Output:**
```json
{
  "MAE": "$101,655",
  "MSE": "26,367,484,252",
  "explained_variance": "81%"
}
```
