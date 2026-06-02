import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
)

st.title("🫀 Heart Disease Predictor")
st.write("Predict the likelihood of heart disease using multiple Machine Learning models.")

age = st.number_input("Age (years)", min_value=0, max_value=150)

sex = st.selectbox("Sex", ["Male", "Female"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=0,
    max_value=300
)

cholesterol = st.number_input(
    "Serum Cholesterol (mm/dl)",
    min_value=0
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
    max_value=202
)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak (ST Depression)",
    min_value=0.0,
    max_value=10.0
)

st_slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)

sex_map = {
    "Male": "M",
    "Female": "F"
}

chest_pain_map = {
    "Typical Angina": "TA",
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "Asymptomatic": "ASY"
}

resting_ecg_map = {
    "Normal": "Normal",
    "ST-T Wave Abnormality": "ST",
    "Left Ventricular Hypertrophy": "LVH"
}

exercise_angina_map = {
    "Yes": "Y",
    "No": "N"
}

st_slope_map = {
    "Upsloping": "Up",
    "Flat": "Flat",
    "Downsloping": "Down"
}

fasting_bs_map = {
    "<= 120 mg/dl": 0,
    "> 120 mg/dl": 1
}

input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [sex_map[sex]],
    'ChestPainType': [chest_pain_map[chest_pain]],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [fasting_bs_map[fasting_bs]],
    'RestingECG': [resting_ecg_map[resting_ecg]],
    'MaxHR': [max_hr],
    'ExerciseAngina': [exercise_angina_map[exercise_angina]],
    'Oldpeak': [oldpeak],
    'ST_Slope': [st_slope_map[st_slope]]
})

algonames = [
    'Decision Trees',
    'Logistic Regression',
    'Random Forest',
    'Support Vector Machine'
]

modelnames = [
    r'artifacts\DecisionTree.pkl',
    r'artifacts\LogisticRegression.pkl',
    r'artifacts\RandomForest.pkl',
    r'artifacts\SVM.pkl'
]

preprocessor = pickle.load(open('artifacts/Preprocessor.pkl', 'rb'))

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
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

    st.write("**Best Model:** Support Vector Machine")
    st.write("**Score:** 90.17%")