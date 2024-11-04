import pandas as pd

def load_and_clean_data(filepath):
    # Load data
    telecom_cust = pd.read_csv(filepath)

    # Convert 'TotalCharges' to a numeric data type, forcing errors to NaN
    telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values (in 'TotalCharges' or any other column)
    telecom_cust.dropna(inplace=True)
    
    # Drop 'customerID' column as it is not needed for analysis
    telecom_cust = telecom_cust.drop(columns=['customerID'])
    
    # Convert the 'Churn' column to binary numeric values
    telecom_cust['Churn'] = telecom_cust['Churn'].map({'Yes': 1, 'No': 0})
    
    # Convert all categorical variables into dummy variables
    df_dummies = pd.get_dummies(telecom_cust)
    
    # Ensure 'Churn' column is retained after dummy variable conversion
    if 'Churn' not in df_dummies.columns:
        raise ValueError("The 'Churn' column is missing from the cleaned data.")
    
    # Save the cleaned data to a CSV file
    df_dummies.to_csv("data/cleaned_data.csv", index=False)
    print("Data cleaned and saved to 'data/cleaned_data.csv'")

if __name__ == "__main__":
    load_and_clean_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
