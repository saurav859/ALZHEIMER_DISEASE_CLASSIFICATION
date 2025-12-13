import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the model and scaler
try:
    model = joblib.load('best_alzheimers_model.pkl')
    scaler = joblib.load('scaler_alzheimers.pkl')
    st.success("Model and scaler loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure 'best_alzheimers_model.pkl' and 'scaler_alzheimers.pkl' are in the same directory.")
    st.stop()

# 2. Define the list of feature names (based on the X DataFrame used during training)
# This list is inferred from the kernel state X.columns
feature_names = [
    'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
    'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
    'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
    'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness'
]

# Define feature types for input widgets (for reference, not used directly in this script but helpful for structuring)
categorical_features = [
    'Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'FamilyHistoryAlzheimers',
    'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
    'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
    'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
]

numerical_features = [
    'Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality',
    'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL',
    'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL'
]

# 3. Streamlit application title
st.title("Alzheimer's Disease Prediction")
st.markdown("---\n### Enter Patient Information for Prediction\n")

# Create input widgets for each feature
user_inputs = {}

# Layout inputs in two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Demographics & Lifestyle")
    user_inputs['Age'] = st.number_input('Age', min_value=60, max_value=100, value=75, step=1)
    user_inputs['Gender'] = st.selectbox('Gender', options={0: 'Female', 1: 'Male'}, format_func=lambda x: {0: 'Female', 1: 'Male'}[x])
    user_inputs['Ethnicity'] = st.selectbox('Ethnicity', options={0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Hispanic'}, format_func=lambda x: {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Hispanic'}[x])
    user_inputs['EducationLevel'] = st.selectbox('Education Level', options={0: 'None', 1: 'High School', 2: 'Bachelors', 3: 'Masters', 4: 'PhD'}, format_func=lambda x: {0: 'None', 1: 'High School', 2: 'Bachelors', 3: 'Masters', 4: 'PhD'}[x])
    user_inputs['BMI'] = st.number_input('BMI', min_value=15.0, max_value=45.0, value=25.0, step=0.1)
    user_inputs['Smoking'] = st.selectbox('Smoking', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['AlcoholConsumption'] = st.number_input('Alcohol Consumption (drinks/week)', min_value=0.0, max_value=30.0, value=5.0, step=0.1)
    user_inputs['PhysicalActivity'] = st.number_input('Physical Activity (hours/week)', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    user_inputs['DietQuality'] = st.number_input('Diet Quality Score (0-10)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    user_inputs['SleepQuality'] = st.number_input('Sleep Quality Score (0-10)', min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    user_inputs['FamilyHistoryAlzheimers'] = st.selectbox('Family History of Alzheimer\'s', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])

with col2:
    st.header("Medical History & Cognitive Scores")
    user_inputs['CardiovascularDisease'] = st.selectbox('Cardiovascular Disease', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['Diabetes'] = st.selectbox('Diabetes', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['Depression'] = st.selectbox('Depression', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['HeadInjury'] = st.selectbox('Head Injury History', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['Hypertension'] = st.selectbox('Hypertension', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['SystolicBP'] = st.number_input('Systolic BP', min_value=90, max_value=200, value=120, step=1)
    user_inputs['DiastolicBP'] = st.number_input('Diastolic BP', min_value=60, max_value=120, value=80, step=1)
    user_inputs['CholesterolTotal'] = st.number_input('Cholesterol Total', min_value=100.0, max_value=300.0, value=200.0, step=1.0)
    user_inputs['CholesterolLDL'] = st.number_input('Cholesterol LDL', min_value=50.0, max_value=200.0, value=100.0, step=1.0)
    user_inputs['CholesterolHDL'] = st.number_input('Cholesterol HDL', min_value=20.0, max_value=100.0, value=50.0, step=1.0)
    user_inputs['CholesterolTriglycerides'] = st.number_input('Cholesterol Triglycerides', min_value=50.0, max_value=300.0, value=150.0, step=1.0)
    user_inputs['MMSE'] = st.number_input('MMSE Score (0-30)', min_value=0, max_value=30, value=25, step=1)
    user_inputs['FunctionalAssessment'] = st.number_input('Functional Assessment Score (0-10)', min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    user_inputs['MemoryComplaints'] = st.selectbox('Memory Complaints', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['BehavioralProblems'] = st.selectbox('Behavioral Problems', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['ADL'] = st.number_input('ADL Score (0-10)', min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    user_inputs['Confusion'] = st.selectbox('Confusion', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['Disorientation'] = st.selectbox('Disorientation', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['PersonalityChanges'] = st.selectbox('Personality Changes', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['DifficultyCompletingTasks'] = st.selectbox('Difficulty Completing Tasks', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
    user_inputs['Forgetfulness'] = st.selectbox('Forgetfulness', options={0: 'No', 1: 'Yes'}, format_func=lambda x: {0: 'No', 1: 'Yes'}[x])


# 4. Create a 'Predict' button
if st.button("Predict Alzheimer's Disease"):
    # 5. Collect user inputs into a pandas DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Ensure the order of columns matches the training data
    input_df = input_df[feature_names]

    # 6. Scale the collected input data
    scaled_input = scaler.transform(input_df)

    # 7. Make a prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)[:, 1]

    # 8. Display the prediction result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"Based on the provided information, the model predicts a HIGH risk of Alzheimer's Disease (Probability: {prediction_proba[0]:.2f})")
    else:
        st.success(f"Based on the provided information, the model predicts a LOW risk of Alzheimer's Disease (Probability: {prediction_proba[0]:.2f})")

    st.write("\n--- Notes ---")
    st.write("0 = Low Risk / No Alzheimer's")
    st.write("1 = High Risk / Alzheimer's")
    st.write("This is a predictive model outcome and should not be used as a substitute for professional medical advice.")
