import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import httpx

app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# URL of the model file
model_url = "https://media.githubusercontent.com/media/MehtaAlok98/Telco-Customer-Churn/refs/heads/main/scripts/churn_model.pkl"

# Load model on startup
model = None

async def fetch_model(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return pickle.loads(response.content)

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = await fetch_model(model_url)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

@dataclass
class InputData:
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

@app.post("/api/predict")
async def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    input_data = [[
        data.SeniorCitizen, data.Partner, data.Dependents, data.tenure,
        data.PhoneService, data.MultipleLines, data.InternetService,
        data.OnlineSecurity, data.OnlineBackup, data.DeviceProtection,
        data.TechSupport, data.StreamingTV, data.StreamingMovies,
        data.Contract, data.PaperlessBilling, data.PaymentMethod,
        data.MonthlyCharges, data.TotalCharges
    ]]
    
    try:
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualize")
async def visualize_data():
    data_url = "https://media.githubusercontent.com/media/MehtaAlok98/Telco-Customer-Churn/refs/heads/main/data/cleaned_data.csv"
    try:
        df = pd.read_csv(data_url)
        if df.empty:
            raise ValueError("Dataset is empty.")
        
        plt.figure(figsize=(10, 6))
        plt.hist(df['Contract'], bins=20, alpha=0.5, label='Contract Type')
        plt.title('Churn Rate by Contract Type')
        plt.xlabel('Contract Type')
        plt.ylabel('Count')
        plt.legend(title='Churn', loc='upper right')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return {"image": f"data:image/png;base64,{image_base64}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
