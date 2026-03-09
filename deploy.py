from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house prices in King County, Washington using a Deep Learning model trained on the KC House dataset.",
    version="1.0.0"
)

with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_weights.pkl", "rb") as f:
    weights = pickle.load(f)

model = Sequential()
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(1))
model.build((None, 20))
model.set_weights(weights)

MAE = 101654.57

class HouseInput(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float
    month: float
    year: float

@app.get('/')
def home():
    return {
        'message': 'Welcome to House Price Prediction API!',
        'endpoints': {
            'POST /predict': 'Send house features to get predicted price',
            'GET /evaluate': 'Get model performance metrics'
        }
    }

@app.post('/predict', summary="Predict House Price", description="Send house features and get the predicted price with a confidence range.")
def predict(data: HouseInput):
    X = np.array(list(data.dict().values())).reshape(1, -1)
    X = scaler.transform(X)
    predicted_price = float(model.predict(X)[0][0])
    return {
        'predicted_price': f"${predicted_price:,.0f}",
        'range': f"Between ${predicted_price - MAE:,.0f} and ${predicted_price + MAE:,.0f}",
        'note': f"Based on model MAE of ${MAE:,.0f}"
    }

@app.get('/evaluate', summary="Model Performance", description="Returns the model evaluation metrics on the test set.")
def evaluate():
    return {
        'MAE': f"${MAE:,.0f}",
        'MSE': f"{26367484251.82:,.0f}",
        'explained_variance': "81%"
    }