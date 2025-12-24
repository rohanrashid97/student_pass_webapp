🎓 Student Pass / Fail Prediction
End-to-End ML Project with Explainable AI & Web App Deployment
📌 Project Overview

This project predicts whether a student will Pass or Fail based on academic and behavioral features.
The project covers the complete machine learning lifecycle:

Data analysis & preprocessing (Google Colab)

Multiple model training & comparison

Best model selection

Explainable AI (XAI) using SHAP

Model saving (.pkl)

Web application deployment using Streamlit

This project was built step-by-step to demonstrate practical ML skills, not just theory.

🧠 Part 1: Work Done in Google Colab (Model Building)
🔹 Dataset

Student Performance Dataset (Kaggle)

CSV file loaded and analyzed in Google Colab

🔹 Data Processing

Loaded dataset using Pandas

Checked shape, columns, data types

Created target variable:

pass_fail = 1 if final score ≥ 10

pass_fail = 0 otherwise

Selected relevant features for modeling

Train–test split

Standardization where needed

🔹 Models Applied

The following models were trained using the same pipeline for fair comparison:

Logistic Regression

Baseline

Hyperparameter tuned (GridSearchCV)

Random Forest

Baseline

Hyperparameter tuned

Support Vector Machine (SVM)

Baseline

Hyperparameter tuned

🔹 Model Evaluation

Each model was evaluated using:

Test Accuracy

Cross-Validation Score

Confusion Matrix

Visual comparison of results

🏆 Best Model Selection
✅ Final Selected Model: Random Forest

Why Random Forest was selected:

Highest test accuracy (~89.9%)

Stable performance between CV and test data

Handles tabular data well

Less sensitive to feature scaling

Better generalization compared to other models

🔍 Part 2: Explainable AI (XAI)

To make the model transparent and trustworthy, Explainable AI was applied.

🔹 Feature Importance

Used Random Forest feature importance

Identified which features influence predictions the most

🔹 SHAP (SHapley Additive exPlanations)

SHAP was used to explain:

Global behavior of the model (summary plot)

Individual predictions (waterfall plot)

What SHAP helped with:

Understanding why the model predicts pass or fail

Explaining decisions at both dataset level and single-student level

Making the model interpretable for non-technical users

💾 Part 3: Model Saving

After selecting the final model:

joblib.dump(best_rf, "final_random_forest_model.pkl")

Why saving the model is important:

Avoid retraining every time

Reuse the trained model

Required for deployment and real-time prediction

🌐 Part 4: Web App Development (Streamlit)

A web application was built so that users can input student data and get predictions.

📁 Project Structure
student_pass_app/
│
├── app.py
└── final_random_forest_model.pkl

🔹 app.py (Core Web App Logic)

Loads the trained .pkl model

Takes user input via form

Converts input to DataFrame

Predicts Pass / Fail

Displays result in browser

🔹 Technologies Used

Streamlit

scikit-learn

Pandas

NumPy

Joblib

🔹 Running the Web App Locally

Install required libraries:

python -m pip install streamlit joblib scikit-learn pandas numpy


Run the app:

python -m streamlit run app.py


Open in browser:

http://localhost:8501

🔹 Common Issues & Fixes

ModuleNotFoundError → install missing library using pip

Blank page → ensure app.py contains code and is saved

Duplicate Streamlit input error → solved using key= parameter

🎯 Final Outcome

✔️ End-to-end ML pipeline implemented

✔️ Best model selected through comparison

✔️ Explainable AI applied using SHAP

✔️ Model saved and reused

✔️ Interactive web application built

🧑‍💻 Skills Demonstrated

Machine Learning model development

Model evaluation & comparison

Explainable AI (XAI)

Python & scikit-learn

Streamlit web app development

Debugging & dependency management

End-to-end project thinking
