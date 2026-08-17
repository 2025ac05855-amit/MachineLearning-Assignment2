# 💰 Adult Income Classification

A Machine Learning web application that predicts whether an individual's annual income belongs to the **<=50K** or **>50K** category.

The application is built using **Python**, **Scikit-learn**, and **Streamlit**. Five trained machine learning models are exported in **PKL format** and deployed through an interactive Streamlit application.

---

## 📌 Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether an individual's income is:

- **<=50K**
- **>50K**

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

```text
income
<=50K
>50K

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
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── random_forest_model.pkl
│   ├── decision_tree_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── gaussian_nb_model.pkl
│   ├── k_nearest_neighbors_model.pkl
│   └── preprocessing.pkl
│
└── notebooks/
    └── AdultIncomePred.ipynb

📦 Trained Models

The trained_models folder contains the trained machine learning models.

random_forest_model.pkl
decision_tree_model.pkl
logistic_regression_model.pkl
gaussian_nb_model.pkl
k_nearest_neighbors_model.pkl

The Streamlit application loads the selected model using joblib.
