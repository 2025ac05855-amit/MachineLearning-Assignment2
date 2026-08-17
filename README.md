# 💰 Adult Income Classification

A Machine Learning web application that predicts whether an individual's annual income belongs to the **<=50K** or **>50K** category.

The application is built using **Python**, **Scikit-learn**, and **Streamlit**. Five trained machine learning models are exported in **PKL format** and deployed through an interactive Streamlit application.

---

## 📌 Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether an individual's income is:

- **<=50K**
- **>50K**

The prediction is based on multiple demographic and employment-related features available in the Adult Income dataset. These features include information such as age, workclass, education, marital status, occupation, relationship, capital gain, capital loss, hours worked per week, and other relevant attributes.

The project addresses the complete machine learning workflow, beginning with data preprocessing and preparation, followed by training multiple classification models and comparing their performance using different evaluation metrics.

The application allows users to:

- Select a machine learning model.
- Upload a CSV file containing input data.
- Generate income predictions.
- View prediction results and class distribution.
- Evaluate the selected model when actual target values are available.
- View Accuracy, AUC, Precision, Recall, F1-Score, and MCC.
- View a Confusion Matrix.
- View a Classification Report.
- Download prediction results as a CSV file.

---

## 📊 Dataset Description

This project uses the **Adult Income dataset**.

The dataset contains demographic and employment-related information used to predict whether an individual's income exceeds 50K.

Typical features include:

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

### Target Variable

income
- **<=50K**
- **>50K**

During training, the target labels are encoded numerically and converted back to their original labels when displaying predictions in the Streamlit application.

🤖 Machine Learning Models

The following five classification models are trained and deployed:

Random Forest
Decision Tree
Logistic Regression
Gaussian Naive Bayes
K-Nearest Neighbors

Each trained model is saved separately in PKL format.

📈 Model Performance

The following performance results were obtained during model evaluation:

ML Model Name	Accuracy	AUC	Precision	Recall	F1-Score	MCC
Logistic Regression	0.842008	0.889641	0.799789	0.738293	0.760729	0.534557
Decision Tree	0.857516	0.905852	0.835108	0.749812	0.778816	0.578668
K-Nearest Neighbors	0.831721	0.855273	0.772971	0.749376	0.759735	0.521813
Gaussian Naive Bayes	0.807616	0.855432	0.766185	0.646830	0.670212	0.395394
Random Forest	0.861201	0.918010	0.835346	0.761603	0.788240	0.592376
Best Performing Model

Based on the reported evaluation metrics, Random Forest achieved the best overall performance:

Highest Accuracy: 0.861201
Highest AUC: 0.918010
Highest F1-Score: 0.788240
Highest MCC: 0.592376

### Project Structure
AdultIncomePrediction/
│
├── AdultIncomePred.ipynb
├── app.py
├── requirements.txt
├── README.md
├── Test_Data.csv
│
└── models/
   ├── random_forest_model.pkl
   ├── decision_tree_model.pkl
   ├── logistic_regression_model.pkl
   ├── gaussian_nb_model.pkl
   ├── k_nearest_neighbors_model.pkl
   └── preprocessing.pkl


📦 Trained Models

The trained_models folder contains the trained machine learning models.

random_forest_model.pkl
decision_tree_model.pkl
logistic_regression_model.pkl
gaussian_nb_model.pkl
k_nearest_neighbors_model.pkl

The Streamlit application loads the selected model using joblib.

⚙️ Preprocessing

The models were trained using preprocessed and scaled input data.

To ensure that uploaded data is processed in exactly the same way during deployment, the preprocessing information is stored in:

preprocessing.pkl

This file contains:

Feature names
Feature order
Numeric columns
Categorical columns
Numeric median values
Category mappings
Target mapping
Inverse target mapping
Fitted scaler
The saved preprocessing artifacts ensure consistency between the training and deployment environments.

🚀 Streamlit Application Features

1. Model Selection
Users can select one of the five available machine learning models from the sidebar.
Random Forest
Decision Tree
Logistic Regression
Gaussian Naive Bayes
K-Nearest Neighbors

2. CSV Upload
Users can upload a CSV file containing the required feature columns.
For prediction: Target column is optional
For evaluation: Include the income target column

3. Income Prediction
The application:
Loads the selected PKL model.
Loads the saved preprocessing artifacts.
Preprocesses the uploaded input data.
Applies the saved scaler.
Generates predictions.
Converts numeric predictions back to income labels.


4. Prediction Summary
The application displays:
Total Predictions
Number of Unique Income Classes
Prediction Distribution
Bar Chart of Predicted Income Classes

5. Model Evaluation
If the uploaded CSV contains the income column, the application calculates:
Accuracy
AUC
Precision (Macro)
Recall (Macro)
F1-Score (Macro)
Matthews Correlation Coefficient (MCC)

6. Confusion Matrix

The application displays the confusion matrix for the selected model.
                    Predicted <=50K    Predicted >50K
Actual <=50K              TN                  FP
Actual >50K               FN                  TP

7. Classification Report

The application displays:

Precision
Recall
F1-Score
Support

for both income classes:

<=50K
>50K

8. Download Predictions

Users can download the prediction results as: adult_income_predictions.csv

The downloaded file includes the original uploaded data and: Predicted Income
