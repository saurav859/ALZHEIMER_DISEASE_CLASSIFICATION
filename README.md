pandas
numpy
matplotlib
seaborn
joblib
scikit-learn
xgboost
streamlit

--
Alzheimer's Disease Prediction with Machine Learning
Project Overview
This project aims to predict Alzheimer's Disease using various patient data, including demographic information, lifestyle factors, medical history, and cognitive scores. The goal is to develop a machine learning model that can assist in early risk assessment, leveraging exploratory data analysis (EDA) and several classification algorithms.

Dataset
The dataset contains 2,149 patient records with 35 features, including:

Demographic: Age, Gender, Ethnicity, Education Level.
Lifestyle: BMI, Smoking, Alcohol Consumption, Physical Activity, Diet Quality, Sleep Quality.
Medical History: Family History of Alzheimer's, Cardiovascular Disease, Diabetes, Depression, Head Injury, Hypertension, Systolic/Diastolic BP, Cholesterol Levels (Total, LDL, HDL, Triglycerides).
Cognitive Scores & Symptoms: MMSE, Functional Assessment, Memory Complaints, Behavioral Problems, ADL, Confusion, Disorientation, Personality Changes, Difficulty Completing Tasks, Forgetfulness.
Target Variable: Diagnosis (0 = No Alzheimer's, 1 = Alzheimer's).
Exploratory Data Analysis (EDA) Key Findings
No major missing values or duplicates were found, indicating a clean dataset.
Irrelevant columns like PatientID and DoctorInCharge were removed.
The target variable Diagnosis showed a slight class imbalance (approximately 65% No Alzheimer's, 35% Alzheimer's), addressed during modeling.
Strongest Predictors: Cognitive scores (MMSE, Functional Assessment, ADL) and symptoms (Confusion, Forgetfulness) showed the most significant correlation with Alzheimer's diagnosis.
Moderate Predictors: Lifestyle and medical factors (BMI, Blood Pressure, Cholesterol, Family History) also played a role.
Some numerical features exhibited outliers, common in medical data, which were noted for potential handling.
Modeling and Evaluation
Several classification models were trained and evaluated:

Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier (Selected Best Model)
XGBoost Classifier
Best Model Selection: The Gradient Boosting Classifier was chosen as the best model due to its superior performance, particularly in terms of Recall (0.914) and ROC-AUC (0.948). It demonstrated a balanced and strong predictive capability with a low rate of False Negatives (missing actual Alzheimer's cases), which is crucial in medical diagnosis.

Model Performance Highlights (Gradient Boosting)
Accuracy: ~94%
Precision: ~91.4%
Recall: ~91.4%
F1 Score: ~91.4%
ROC-AUC: ~0.948
