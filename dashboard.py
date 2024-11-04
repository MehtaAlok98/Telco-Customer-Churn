# dashboard.py
import streamlit as st
import pandas as pd
import requests

st.title("Customer Churn Prediction Dashboard")

st.write("### Enter Customer Data for Prediction")
feature1 = st.number_input("Feature 1", min_value=0.0)
feature2 = st.number_input("Feature 2", min_value=0.0)
# Add more input fields as required

if st.button("Predict"):
    response = requests.post("https://your-vercel-url.vercel.app/predict", json={"feature1": feature1, "feature2": feature2})
    st.write("Prediction:", response.json()["prediction"])
