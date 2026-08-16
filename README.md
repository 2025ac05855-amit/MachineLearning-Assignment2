# Adult Income Classification – Machine Learning Project

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for predicting an individual's income class using demographic and employment-related information from the Adult Income dataset.

The project evaluates different classification algorithms under the same preprocessing and testing conditions. The models are compared using Accuracy, AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).

The goal is to identify the model that provides the best overall predictive performance for the selected dataset and can subsequently be used for deployment through a Streamlit application.


## b. Dataset Description

The **Adult Income dataset** is a binary classification dataset containing demographic and employment-related information about individuals.

The target variable represents the individual's income category.

The dataset contains both numerical and categorical features. Therefore, preprocessing is required before applying the machine learning models.

### Data Preprocessing

The following preprocessing steps were performed:

1. Numerical and categorical columns were identified.
2. Missing numerical values were handled using median imputation.
3. Categorical variables were converted into numerical values using category mappings.
4. Unknown categorical values during deployment are assigned a value of `-1`.
5. Features were standardized using the training-data mean and scale.
6. The dataset was divided into training and testing sets.
7. The same preprocessing parameters used during training were saved for deployment.


## c. GitHub Repository Link

The complete project, including the Jupyter Notebook, README, requirements file, Streamlit application and trained model, is maintained in the following GitHub repository:

**GitHub Repository:**
https://github.com/2025ac05855-amit/MachineLearning-Assignment2/
The repository contains the following important files:
- 2025AC05855 - AdultIncomePred.ipynb
- README.md
- requirements.txt
- test_data.csv
- app.py
- models/

The `models` directory contains the exported deployment-safe model modules and preprocessing information.


## d. Models Used

The following five machine learning classification models were implemented and evaluated:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)



## Model Evaluation Metrics

The models were evaluated using the following metrics:

* **Accuracy** – Measures the overall proportion of correctly classified observations.
* **AUC** – Measures the model's ability to distinguish between the two classes across different classification thresholds.
* **Precision** – Measures the proportion of predicted positive instances that are actually positive.
* **Recall** – Measures the proportion of actual positive instances correctly identified by the model.
* **F1-Score** – Provides a balance between Precision and Recall.
* **MCC** – Matthews Correlation Coefficient provides a balanced measure of binary classification performance, particularly when class distributions are unequal.

### Comparison of Model Performance

| ML Model Name            |   Accuracy   |     AUC      |   Precision  |   Recall     |     F1       |    MCC       |
| ------------------------ |  ----------: |  ----------: | -----------: | -----------: | -----------: | -----------: |
| Logistic Regression      | **0.842008** | **0.889641** | **0.799789** | **0.738293** | **0.760729** | **0.534557** |
| Decision Tree            | **0.857516** | **0.905852** | **0.835108** | **0.749812** | **0.778816** | **0.578668** |
| kNN                      | **0.831721** | **0.855273** | **0.772971** | **0.749376** | **0.759735** | **0.521813** |
| Naive Bayes              | **0.807616** | **0.855432** | **0.766185** | **0.646830** | **0.670212** | **0.395394** |
| Random Forest (Ensemble) | **0.861201** | **0.918010** | **0.835346** | **0.761603** | **0.788240** | **0.592376** |


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

# Project Files
MachineLearning-Assignment2/
│
├── 2025AC05855-AdultIncomePred.ipynb
├── README.md
├── requirements.txt
├── app.py
├── test_data.csv
│
└── python_models/
    ├── random_forest_model.py
    ├── gaussian_nb_model.py
    ├── k_nearest_neighbors_model.py
    ├── decision_tree_model.py
    ├── logistic_regression_model.py
    └── feature_names.txt

# Requirements

The required Python packages are listed in `requirements.txt`:
•	streamlit
•	pandas
•	numpy
•	scikit-learn
•	matplotlib
•	seaborn
Install the dependencies using:pip install -r requirements.txt
