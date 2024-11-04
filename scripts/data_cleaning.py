# scripts/data_cleaning.py
import pandas as pd

def load_and_clean_data(filepath):
    # Load data
    data = pd.read_csv(filepath)
    
    # Drop 'customerID' as it is not needed for analysis
    data = data.drop(columns=['customerID'])
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Convert categorical columns to dummy variables, excluding the 'Churn' column
    # data = pd.get_dummies(data, columns=data.select_dtypes(include=['object']).columns.difference(['Churn']), drop_first=True)
    
    # Ensure 'Churn' column is retained
    if 'Churn' not in data.columns:
        raise ValueError("The 'Churn' column is missing from the cleaned data.")
    
    # Save cleaned data
    data.to_csv("data/cleaned_data.csv", index=False)
    print("Data cleaned and saved to 'data/cleaned_data.csv'")

if __name__ == "__main__":
    load_and_clean_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
