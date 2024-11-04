# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("scripts/churn_model.pkl")

class InputData(BaseModel):
    feature1: float
    feature2: float
    # Add additional features here

@app.post("/predict")
async def predict(data: InputData):
    input_data = [[data.feature1, data.feature2]]  # Adjust according to features
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
