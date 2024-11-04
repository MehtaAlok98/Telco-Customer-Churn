# scripts/data_cleaning.py
import pandas as pd

def load_and_clean_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    
    # Clean and preprocess
    data = data.dropna()  # Example cleaning step
    data = pd.get_dummies(data, drop_first=True)  # Convert categorical to numeric

    # Ensure 'Churn' column is retained and not dropped
    if 'Churn' not in data.columns:
        raise ValueError("The 'Churn' column is missing from the cleaned data.")
    
    # Save cleaned data
    data.to_csv("data/cleaned_data.csv", index=False)
    print("Data cleaned and saved to 'data/cleaned_data.csv'")

if __name__ == "__main__":
    load_and_clean_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
