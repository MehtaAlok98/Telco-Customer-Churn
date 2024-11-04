from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import requests

app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model from the URL
model_url = "https://media.githubusercontent.com/media/MehtaAlok98/Telco-Customer-Churn/refs/heads/main/scripts/churn_model.pkl"
response = requests.get(model_url)
with open("churn_model.pkl", "wb") as f:
    f.write(response.content)
model = joblib.load("churn_model.pkl")

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

@app.get("/visualize")
async def visualize_data():
    try:
        # Load the dataset from the URL
        data_url = "https://media.githubusercontent.com/media/MehtaAlok98/Telco-Customer-Churn/refs/heads/main/data/cleaned_data.csv"
        df = pd.read_csv(data_url)  

        if df.empty:
            raise ValueError("Dataset is empty.")

        plt.figure(figsize=(10, 6))
        sns.countplot(x='Contract', hue='Churn', data=df)
        plt.title('Churn Rate by Contract Type')
        plt.xlabel('Contract Type')
        plt.ylabel('Count')
        plt.legend(title='Churn', loc='upper right')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return {"image": f"data:image/png;base64,{image_base64}"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
