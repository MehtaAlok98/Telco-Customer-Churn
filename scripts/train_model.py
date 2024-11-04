# scripts/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    # Load cleaned data
    data = pd.read_csv("data/cleaned_data.csv")
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, "scripts/churn_model.pkl")
    print("Model trained and saved as 'scripts/churn_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()
