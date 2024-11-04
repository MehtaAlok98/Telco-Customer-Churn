# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("scripts/churn_model.pkl")

class InputData(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
async def predict(data: InputData):
    input_data = [[
        data.SeniorCitizen, data.Partner, data.Dependents, data.tenure, 
        data.PhoneService, data.MultipleLines, data.InternetService,
        data.OnlineSecurity, data.OnlineBackup, data.DeviceProtection,
        data.TechSupport, data.StreamingTV, data.StreamingMovies,
        data.Contract, data.PaperlessBilling, data.PaymentMethod,
        data.MonthlyCharges, data.TotalCharges
    ]]
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
