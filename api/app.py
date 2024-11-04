import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass
import pickle
import httpx
from collections import Counter

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL of the model file and dataset
model_url = "https://media.githubusercontent.com/media/MehtaAlok98/Telco-Customer-Churn/refs/heads/main/scripts/churn_model.pkl"
data_url = "https://media.githubusercontent.com/media/MehtaAlok98/Telco-Customer-Churn/refs/heads/main/data/cleaned_data.csv"

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

@app.get("/api/data-summary")
async def data_summary():
    try:
        # Fetch the dataset
        async with httpx.AsyncClient() as client:
            response = await client.get(data_url)
            response.raise_for_status()
            data = response.text.splitlines()
        
        # Process the data
        headers = data[0].split(",")  # Extract headers from the first line
        contract_idx = headers.index("Contract")
        churn_idx = headers.index("Churn")

        contract_counts = Counter()
        churn_counts = Counter()

        for row in data[1:]:
            fields = row.split(",")
            contract_counts[fields[contract_idx]] += 1
            churn_counts[fields[churn_idx]] += 1

        # Prepare summary data
        summary = {
            "contract_counts": dict(contract_counts),
            "churn_counts": dict(churn_counts),
        }

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
