import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import json
from pathlib import Path

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
)

st.title("🫀 Heart Disease Predictor")
st.write("Predict the likelihood of heart disease using multiple Machine Learning models.")

age = st.number_input("Age (years)", min_value=0, max_value=150, value=45)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=0,
    max_value=300,
    value=120
)

cholesterol = st.number_input(
    "Serum Cholesterol (mm/dl)",
    min_value=0,
    value=193
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    ["<= 120 mg/dl", "> 120 mg/dl"]
)

resting_ecg = st.selectbox(
    "Resting ECG Results",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
)

max_hr = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=202,
    value=150
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

st_slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)


input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chest_pain],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs],
    'RestingECG': [resting_ecg],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope]
})

algonames = [
    'Decision Trees',
    'Logistic Regression',
    'Random Forest',
    'Support Vector Machine'
]

modelnames = [
    Path('artifacts/DecisionTree.pkl'),
    Path('artifacts/LogisticRegression.pkl'),
    Path('artifacts/RandomForest.pkl'),
    Path('artifacts/SVM.pkl')
]

preprocessor = pickle.load(open(Path('artifacts/Preprocessor.pkl'), 'rb'))

# Load the dynamic metrics
with open(Path('artifacts/metrics.json'), 'r') as f:
    model_metrics = json.load(f)

def predict_heart_disease(data):
    predictions = []

    processed_data = preprocessor.transform(data)

    for modelname in modelnames:
        model = pickle.load(open(modelname, 'rb'))
        prediction = model.predict(processed_data)
        predictions.append(prediction)

    return predictions


if st.button("Submit"):

    result = predict_heart_disease(input_data)

    st.subheader("Model Predictions")

    prediction_data = []

    for i in range(len(result)):

        prediction = (
            "Disease"
            if result[i][0] == 1
            else "No Disease"
        )

        prediction_data.append({
            "Model": algonames[i],
            "Prediction": prediction
        })

    prediction_df = pd.DataFrame(prediction_data)

    st.table(prediction_df)

    st.markdown("---")

    svm_index = algonames.index("Support Vector Machine")
    final_prediction = result[svm_index][0]

    st.subheader("Final Recommendation")

    if final_prediction == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")


    st.write("**Best Model:** Support Vector Machine")

    svm_acc = model_metrics["SVM"]["Accuracy"] * 100 
    st.write(f"**Score:** {svm_acc:.2f}%")

    st.markdown("---")

    
