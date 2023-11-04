# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("classification_api")

# Create input/output pydantic models
input_model = create_model("classification_api_input", **{'Marital Status': 1.0, 'Application mode': 43.0, 'Application order': 1.0, 'Course': 9238.0, 'Daytime/evening attendance': 1.0, 'Previous qualification': 39.0, 'Previous qualification (grade)': 120.0, 'Nacionality': 1.0, "Mother's qualification": 2.0, "Father's qualification": 1.0, "Mother's occupation": 1.0, "Father's occupation": 2.0, 'Admission grade': 118.5999984741211, 'Displaced': 0.0, 'Educational special needs': 0.0, 'Debtor': 0.0, 'Tuition fees up to date': 1.0, 'Gender': 1.0, 'Scholarship holder': 0.0, 'Age at enrollment': 21.0, 'International': 0.0, 'Curricular units 1st sem (credited)': 9.0, 'Curricular units 1st sem (enrolled)': 11.0, 'Curricular units 1st sem (evaluations)': 14.0, 'Curricular units 1st sem (approved)': 10.0, 'Curricular units 1st sem (grade)': 12.399999618530273, 'Curricular units 1st sem (without evaluations)': 0.0, 'Curricular units 2nd sem (credited)': 6.0, 'Curricular units 2nd sem (enrolled)': 10.0, 'Curricular units 2nd sem (evaluations)': 13.0, 'Curricular units 2nd sem (approved)': 10.0, 'Curricular units 2nd sem (grade)': 11.800000190734863, 'Curricular units 2nd sem (without evaluations)': 0.0, 'Unemployment rate': 13.899999618530273, 'Inflation rate': -0.30000001192092896, 'GDP': 0.7900000214576721})
output_model = create_model("classification_api_output", prediction='Dropout')


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
