from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np
import logging
import os

app = FastAPI()

# Define paths to the models
model_path = os.path.join(os.path.dirname(__file__), "models/logistic_classifier_best.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "models/robust_scaler.pkl")

# Load the machine learning model
model = joblib.load(model_path)

# Load the scaler
scaler = joblib.load(scaler_path)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define the input data schema with validation
class CustomerData(BaseModel):
    age: int = Field(..., ge=0, description="Age of the client, must be non-negative")
    job: Literal[
        'admin.', 'blue-collar', 'technician', 'services',
        'management', 'retired', 'self-employed', 'entrepreneur',
        'unemployed', 'housemaid', 'student', 'unknown'
    ] = Field(..., description="Job type")
    marital: Literal[
        'married', 'single', 'divorced', 'unknown'
    ] = Field(..., description="Marital status")
    education: Literal[
        'university.degree', 'high.school', 'basic.9y', 'professional.course',
        'basic.4y', 'basic.6y', 'unknown', 'illiterate'
    ] = Field(..., description="Education level")
    default: Literal[
        'yes', 'no', 'unknown'
    ] = Field(..., description="Has credit in default?")
    housing: Literal[
        'yes', 'no', 'unknown'
    ] = Field(..., description="Has housing loan?")
    loan: Literal[
        'yes', 'no', 'unknown'
    ] = Field(..., description="Has personal loan?")
    contact: Literal[
        'cellular', 'telephone'
    ] = Field(..., description="Contact communication type")
    month: Literal[
        'jan', 'feb', 'mar', 'apr',
        'may', 'jun', 'jul', 'aug',
        'sep', 'oct', 'nov', 'dec'
    ] = Field(..., description="Last contact month of year")
    day_of_week: Literal[
        'mon', 'tue', 'wed', 'thu', 'fri'
    ] = Field(..., description="Last contact day of the week")
    duration: int = Field(..., ge=0, description="Last contact duration, in seconds, must be non-negative")
    campaign: int = Field(..., ge=0, description="Number of contacts performed during this campaign and for this client, must be non-negative")
    pdays: int = Field(..., ge=0, description="Number of days since the client was last contacted (999 = never contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts performed before this campaign")
    poutcome: Literal[
        'nonexistent', 'failure', 'success'
    ] = Field(..., description="Outcome of the previous marketing campaign")

# Define label encoders and mappings
label_education = {'education': {'university.degree': 7, 'high.school': 5, 'basic.9y': 4,
                                 'professional.course': 6, 'basic.4y': 2, 'basic.6y': 3,
                                 'unknown': 0, 'illiterate': 1}}
label_month = {'month': {'jan': 0, 'feb': 1, 'mar': 2,
                         'apr': 3, 'may': 4, 'jun': 5,
                         'jul': 6, 'aug': 7, 'sep': 8,
                         'oct': 9, 'nov': 10, 'dec': 11}}
label_day_of_week = {'day_of_week': {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}}
binary_encode_contact = {'contact': {'cellular': 1, 'telephone': 0}}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Subscription Prediction API"}

@app.post("/predict")
def predict(customer_data: CustomerData):
    data = customer_data.dict()

    # Log raw input data
    logging.info(f"Raw input data: {data}")

    # Convert input data to DataFrame
    df_input = pd.DataFrame([data])

    # Replace categorical values with label encodings and binary encodings
    df_input.replace(label_education, inplace=True)
    df_input.replace(label_month, inplace=True)
    df_input.replace(label_day_of_week, inplace=True)
    df_input.replace(binary_encode_contact, inplace=True)

    # Perform one-hot encoding for the remaining categorical features
    df_input = pd.get_dummies(df_input, columns=['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

    # Ensure all columns are present in the input data
    missing_cols = set(model.feature_names_in_) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    df_input = df_input[model.feature_names_in_]

    # Log processed input data before scaling
    logging.info(f"Processed input data before scaling: {df_input}")

    # Scale numeric features
    numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous']
    df_input[numeric_columns] = scaler.transform(df_input[numeric_columns])

    # Log processed and scaled input data
    logging.info(f"Processed input data after scaling: {df_input}")

    # Predict using the loaded model
    prediction = model.predict(df_input)

    # Map prediction to "yes" or "no"
    prediction_label = "yes" if prediction[0] == 1 else "no"

    # Log prediction
    logging.info(f"Prediction: {prediction_label}")

    return {"prediction": prediction_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
