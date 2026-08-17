# 💰 Adult Income Classification
A Machine Learning web application that predicts whether an individual's annual income belongs to the **<=50K** or **>50K** category.
The prediction is based on multiple demographic and employment-related features available in the Adult Income dataset. These features include information such as age, workclass, education, marital status, occupation, relationship, capital gain, capital loss, hours worked per week, and other relevant attributes.
The application is built using **Python**, **Scikit-learn**, and **Streamlit**. Five trained machine learning models are exported in **PKL format** and deployed through an interactive Streamlit application.

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

**🤖 Machine Learning Models**
The following five classification models are trained and deployed:
Random Forest
Decision Tree
Logistic Regression
Gaussian Naive Bayes
K-Nearest Neighbors
Each trained model is saved separately in PKL format.

**📈 Model Performance**
The following performance results were obtained during model evaluation:
ML Model Name	            Accuracy	      AUC      Precision	   Recall	   F1	         MCC
Logistic Regression	      0.842008	   0.889641	   0.799789	      0.738293	   0.760729	   0.534557
Decision Tree	            0.857516	   0.905852	   0.835108	      0.749812	   0.778816	   0.578668
kNN	                     0.831721	   0.855273	   0.772971	      0.749376	   0.759735	   0.521813
Naïve Bayes	               0.807616	   0.855432	   0.766185	      0.646830	   0.670212	   0.395394
Random Forest	            0.861201	   0.918010	   0.835346	      0.761603	   0.788240	   0.592376

# Observations on Model Performance
## Logistic Regression
Logistic Regression achieved 84.20% accuracy and an AUC of 0.8896. It provides a strong baseline and performs reasonably well, but its linear decision boundary limits its ability to capture complex non-linear relationships in the dataset.
Logistic Regression provides a strong baseline for the Adult Income classification problem. Since it is a linear classification algorithm, it performs well when the relationship between the processed features and the target can be reasonably represented by a linear decision boundary.
Its performance provides a useful benchmark against the more complex tree-based and instance-based models.
Logistic Regression provides a competitive baseline, but its ability to capture complex non-linear relationships is more limited than ensemble and tree-based approaches.
## Decision Tree
Decision Tree achieved 85.75% accuracy, 0.9059 AUC, and 0.7788 F1-score. It performs better than Logistic Regression and can capture non-linear relationships and feature interactions. However, as a single tree, it is generally more susceptible to overfitting than an ensemble model.
The Decision Tree model is capable of learning non-linear relationships and feature interactions. This makes it suitable for the Adult Income dataset, which contains relationships between demographic and employment characteristics.
However, a single decision tree can be sensitive to the training data and may not generalize as effectively as an ensemble method.
The Decision Tree captures non-linear patterns effectively but may have lower generalization performance compared with Random Forest.
## k-Nearest Neighbors (kNN)
kNN achieved 83.17% accuracy and 0.8553 AUC. Its recall of 0.7494 is comparable to the Decision Tree, but its overall accuracy, AUC, F1-score, and MCC are lower. The model benefits from feature scaling because it relies on distance calculations.
The kNN model classifies an observation based on the classes of nearby observations in the scaled feature space.
Feature scaling is particularly important for kNN because the algorithm relies on distance calculations.
kNN can provide reasonable classification performance after feature scaling, but its performance depends strongly on the distribution of the data and the choice of neighborhood size. It can also be computationally more expensive during prediction for larger datasets.
## Naive Bayes
Naive Bayes produced the lowest overall performance, with 80.76% accuracy, 0.8554 AUC, 0.6702 F1-score, and 0.3954 MCC. Its relatively low recall of 0.6468 indicates weaker identification of the positive class. Its simplifying independence assumptions may not adequately represent relationships between the dataset's features.
Gaussian Naive Bayes assumes that the numerical features follow Gaussian distributions within each class and that features are conditionally independent given the target class. This makes the model computationally efficient and provides a useful probabilistic baseline.
Naive Bayes is computationally efficient and provides a useful baseline, although its simplifying independence assumptions may limit its ability to model relationships between correlated features.
## Random Forest (Ensemble)
Random Forest achieved the best performance across every reported metric: 86.12% accuracy, 0.9180 AUC, 0.8353 precision, 0.7616 recall, 0.7882 F1-score, and 0.5924 MCC. Its ensemble structure allows it to capture complex non-linear relationships while improving generalization compared with a single Decision Tree.
Random Forest combines multiple decision trees to improve predictive performance and generalization. The model used in this project consists of multiple trees with controlled tree depth and minimum leaf size:
RandomForestClassifier(
    random_state=42,
    max_depth=10,
    min_samples_leaf=4,
    n_estimators=40
)
Because multiple trees contribute to the final prediction, Random Forest is generally more robust than an individual Decision Tree.
Random Forest is expected to provide strong overall performance because it can model complex non-linear relationships and feature interactions while reducing the risk of overfitting associated with a single decision tree.

# Overall Winner for the Dataset

The **Overall Winner** should be selected based on the actual evaluation results rather than assuming a model in advance.

The model with the strongest overall combination of **Accuracy, AUC, Precision, Recall, F1-Score, and MCC** should be identified as the final winner.

### Overall Winner

**Random Forest (Ensemble)** is the overall winner.
It achieved the highest value for all six evaluation metrics:
Accuracy: 0.861201
AUC: 0.918010
Precision: 0.835346
Recall: 0.761603
F1: 0.788240
MCC: 0.592376
Therefore, Random Forest provides the strongest overall predictive performance for this Adult Income classification dataset.

### Project Structure
AdultIncomePrediction/
│
├── 2025AC05855 - AdultIncomePred.ipynb
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
To ensure that uploaded data is processed in exactly the same way during deployment, the preprocessing information is stored in preprocessing.pkl
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
for both income classes: <=50K and >50K

8. Download Predictions
Users can download the prediction results as: adult_income_predictions.csv
The downloaded file includes the original uploaded data and Predicted Income
