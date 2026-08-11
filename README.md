# MachineLearning-Assignment2
**Problem Statement:**
Implement the following classification models using the dataset chosen above. All the 5 ML models have to be implemented on the same dataset.
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Naive Bayes Classifier - Gaussian or Multinomial
  5. Ensemble Model - Random Forest

**Dataset Description: **
I'm using Binary Classification dataset called Adult Income. Data is downloaded from Kaggle.
Adult Income Classification – Machine Learning Model Deployment
Overview

This project builds and evaluates multiple machine learning classification models to predict whether an individual's annual income is above or below the specified income threshold using the Adult Income dataset.

The project includes:

Data preprocessing and feature engineering
Handling of numerical and categorical features
Missing-value treatment
Feature scaling
Training and evaluation of multiple classification models
Performance comparison using multiple evaluation metrics
Export of deployment-safe Python model modules
Export of preprocessing metadata
Streamlit-based model deployment
Models Used

The following five classification algorithms are trained and evaluated:

Random Forest
Gaussian Naive Bayes
K-Nearest Neighbors (KNN)
Decision Tree
Logistic Regression
Model Configuration
Model	Configuration
Random Forest	n_estimators=40, max_depth=10, min_samples_leaf=4
Gaussian NB	Default configuration
KNN	Default configuration
Decision Tree	max_depth=8, min_samples_leaf=4
Logistic Regression	max_iter=1000

A random seed of 42 is used where applicable to improve reproducibility.

Dataset

The project uses the Adult Income dataset, which contains demographic and employment-related information.

The target variable represents the income class.

Feature Types

The preprocessing pipeline separates features into:

Numerical columns
Categorical columns

Categorical variables are converted into numerical representations using the category mappings generated during preprocessing.

Preprocessing

The same preprocessing logic used during model training is exported with the deployment modules to ensure that predictions made by the deployed application use the same transformations as the training pipeline.

Numerical Features

Numerical columns are:

Converted to numeric values.
Invalid values are treated as missing.
Missing values are replaced using the training-data median.
Features are standardized using the training scaler.

The standardization formula is:

scaled_value = (value - mean) / scale

The scaler means and scales are stored in:

python_models/preprocessing_metadata.json
Categorical Features

Categorical columns are:

Converted to string values.
Missing values are handled.
Leading and trailing whitespace is removed.
Training-time category mappings are applied.
Unknown categories are assigned -1.

This ensures that the deployment preprocessing remains consistent with the training preprocessing.

Model Evaluation

Each model is evaluated on the test dataset using the following metrics:

Accuracy
Precision (Macro)
Recall (Macro)
F1-Score (Macro)
ROC AUC
Matthews Correlation Coefficient (MCC)

The resulting metrics are saved to:

python_models/model_performance_metrics.csv
Performance Comparison

A comparison plot can be generated to visually compare the performance of all models.

The plot contains numerical labels for each metric value.

Output file:

python_models/model_performance_comparison.png
Project Structure
project/
│
├── AdultIncomePred.ipynb
│
├── README.md
│
├── python_models/
│   ├── random_forest_model.py
│   ├── gaussian_nb_model.py
│   ├── k_nearest_neighbors_model.py
│   ├── decision_tree_model.py
│   ├── logistic_regression_model.py
│   │
│   ├── preprocessing_metadata.json
│   ├── feature_names.txt
│   ├── model_performance_metrics.csv
│   └── model_performance_comparison.png
│
└── streamlit_app.py

The exact files may vary depending on the final project configuration.

Exported Model Modules

The trained models are converted into standalone Python modules.

For models supported by m2cgen, the generated source code contains the model's prediction logic directly in Python.

For example:

result = score(scaled_array)

The original m2cgen score() function is preserved to maintain compatibility with the Streamlit deployment.

Each exported module also provides:

predict(data)

and:

predict_labels(data)
Example
from python_models.logistic_regression_model import predict_labels

prediction = predict_labels(input_data)

print(prediction)
Deployment-Safe Prediction Pipeline

The exported model modules follow this pipeline:

User Input
    │
    ▼
Input Validation
    │
    ▼
Missing Value Handling
    │
    ▼
Categorical Encoding
    │
    ▼
Feature Ordering
    │
    ▼
Feature Scaling
    │
    ▼
Trained Model
    │
    ▼
Prediction
    │
    ▼
Original Target Label

This allows the Streamlit application to make predictions without retraining the models.

Streamlit Application

The trained models can be integrated into a Streamlit application for interactive prediction.

The application can:

Accept user inputs
Apply the same preprocessing used during training
Load the exported model module
Generate predictions
Display the predicted income class
Compare available models

A typical Streamlit execution command is:

streamlit run streamlit_app.py
Installation

Install the required Python packages using:

pip install numpy pandas scikit-learn matplotlib m2cgen streamlit

If a requirements.txt file is provided, install dependencies using:

pip install -r requirements.txt
Running the Notebook

Open the notebook:

AdultIncomePred.ipynb

Run the cells in order.

The notebook will:

Load the dataset.
Explore the data.
Identify numerical and categorical columns.
Handle missing values.
Encode categorical variables.
Scale the features.
Split the data into training and testing sets.
Train the five classification models.
Calculate evaluation metrics.
Export the trained model logic.
Export preprocessing metadata.
Generate the model performance CSV.
Generate the performance comparison plot.
Reproducibility

Random state 42 is used for models where the parameter is available.

For example:

RandomForestClassifier(
    random_state=42,
    max_depth=10,
    min_samples_leaf=4,
    n_estimators=40
)

and:

DecisionTreeClassifier(
    random_state=42,
    max_depth=8,
    min_samples_leaf=4
)

This helps produce consistent results when the same dataset and preprocessing configuration are used.

Model Selection

The models should be compared using all reported metrics rather than Accuracy alone.

Particular attention should be given to:

F1-Score (Macro) – useful when balanced performance across classes is important.
ROC AUC – measures the model's ability to distinguish between the two classes.
MCC – provides a balanced evaluation of binary classification performance.
Precision and Recall – provide additional insight into class-specific prediction behavior.

The final model can be selected based on the requirements of the application and the overall evaluation results.

Generated Files
preprocessing_metadata.json

Contains the information required to reproduce preprocessing during deployment, including:

Feature names
Numerical columns
Categorical columns
Numerical medians
Category mappings
Scaler means
Scaler scales
Target mapping
Inverse target mapping
feature_names.txt

Contains the final feature order used by the models.

model_performance_metrics.csv

Contains the evaluation results for all five models.

Example structure:

Model,Accuracy,Precision (Macro),Recall (Macro),F1-Score (Macro),ROC AUC,MCC,Module_Key
Model Python Files

Each model has its own deployment module:

random_forest_model.py
gaussian_nb_model.py
k_nearest_neighbors_model.py
decision_tree_model.py
logistic_regression_model.py

These modules contain the model prediction logic and preprocessing functions required for deployment.

Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
m2cgen
Streamlit
Jupyter Notebook
Key Features
Consistent Preprocessing

The preprocessing information used during training is exported and reused during deployment.

Multiple Model Comparison

Five different classification algorithms are trained and evaluated under the same test conditions.

Deployment-Ready Models

The trained models are converted into standalone Python modules that can be used by the Streamlit application.

Reproducible Evaluation

The project stores model performance metrics in a CSV file, allowing results to be easily reviewed or compared.

Interactive Deployment

The exported modules can be integrated into a Streamlit interface for real-time predictions.

Author

Amit Krishna

Machine Learning Classification and Deployment Project

**GITHUB Repository Link:** https://github.com/2025ac05855-amit/MachineLearning-Assignment2/tree/main

**Models Used:**
ML Model Used            Accuracy    AUC    Precision    Recall    F1      MCC
Logistic Regression
Decision Tree
KNN
Gaussian NB
Random Forest

**Observations:**
Logistic Regression
Decision Tree
KNN
Gaussian NB
Random Forest
