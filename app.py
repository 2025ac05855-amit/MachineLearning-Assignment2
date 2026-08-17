# ============================================================
# STREAMLIT APP - ADULT INCOME CLASSIFICATION
# PKL MODEL DEPLOYMENT
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_DIR = APP_DIR / "models"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "Random Forest":
        MODEL_DIR / "random_forest_model.pkl",

    "Decision Tree":
        MODEL_DIR / "decision_tree_model.pkl",

    "Logistic Regression":
        MODEL_DIR / "logistic_regression_model.pkl",

    "Gaussian Naive Bayes":
        MODEL_DIR / "gaussian_nb_model.pkl",

    "K-Nearest Neighbors":
        MODEL_DIR / "k_nearest_neighbors_model.pkl",
}


# ============================================================
# PREPROCESSING FILE
# ============================================================

PREPROCESSING_PATH = (
    MODEL_DIR / "preprocessing.pkl"
)


# ============================================================
# TARGET CONFIGURATION
# ============================================================

TARGET_COLUMN = "income"


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

@st.cache_resource
def load_preprocessing_artifacts():
    """
    Load the preprocessing information saved during training.

    Expected artifacts:
        feature_names
        numeric_columns
        categorical_columns
        numeric_medians
        category_mappings
        target_mapping
        inverse_target_mapping
        scaler
    """

    if not PREPROCESSING_PATH.exists():

        raise FileNotFoundError(
            "Preprocessing file not found:\n"
            f"{PREPROCESSING_PATH}"
        )

    artifacts = joblib.load(
        PREPROCESSING_PATH
    )

    required_keys = [
        "feature_names",
        "numeric_columns",
        "categorical_columns",
        "numeric_medians",
        "category_mappings",
        "target_mapping",
        "inverse_target_mapping",
        "scaler"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in artifacts
    ]

    if missing_keys:

        raise ValueError(
            "preprocessing.pkl is missing "
            f"required keys: {missing_keys}"
        )

    return artifacts


# ============================================================
# LOAD SELECTED PKL MODEL
# ============================================================

@st.cache_resource
def load_model(model_name):
    """
    Load the selected trained PKL model.
    """

    if model_name not in MODEL_CONFIG:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_path = MODEL_CONFIG[
        model_name
    ]

    if not model_path.exists():

        raise FileNotFoundError(
            "Model file not found:\n"
            f"{model_path}"
        )

    model = joblib.load(
        model_path
    )

    return model


# ============================================================
# PREPROCESS INPUT DATA
# ============================================================

def preprocess_input(
    input_data,
    artifacts
):
    """
    Applies the same preprocessing used during training.

    Steps:
        1. Check required columns
        2. Arrange columns in training order
        3. Convert numeric columns
        4. Fill missing numeric values
        5. Encode categorical columns
        6. Apply the saved StandardScaler
    """

    df = input_data.copy()

    feature_names = artifacts[
        "feature_names"
    ]

    numeric_columns = artifacts[
        "numeric_columns"
    ]

    categorical_columns = artifacts[
        "categorical_columns"
    ]

    numeric_medians = artifacts[
        "numeric_medians"
    ]

    category_mappings = artifacts[
        "category_mappings"
    ]

    scaler = artifacts[
        "scaler"
    ]

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in feature_names
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "The uploaded CSV is missing "
            "required feature columns:\n"
            + str(missing_columns)
        )

    # --------------------------------------------------------
    # KEEP ONLY FEATURES USED DURING TRAINING
    # --------------------------------------------------------

    df = df[
        feature_names
    ].copy()

    # --------------------------------------------------------
    # NUMERIC PREPROCESSING
    # --------------------------------------------------------

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        median_value = numeric_medians.get(
            column
        )

        if median_value is None:

            raise ValueError(
                f"Missing median value for "
                f"numeric column: {column}"
            )

        df[column] = df[column].fillna(
            median_value
        )

    # --------------------------------------------------------
    # CATEGORICAL PREPROCESSING
    # --------------------------------------------------------

    for column in categorical_columns:

        if column not in category_mappings:

            raise ValueError(
                f"Missing category mapping for "
                f"column: {column}"
            )

        mapping = category_mappings[
            column
        ]

        series = (
            df[column]
            .astype("string")
            .fillna("__MISSING__")
            .str.strip()
        )

        df[column] = (
            series
            .map(mapping)
            .fillna(-1)
            .astype(float)
        )

    # --------------------------------------------------------
    # CONVERT ALL FEATURES TO NUMERIC
    # --------------------------------------------------------

    X_processed = df.astype(
        float
    )

    # --------------------------------------------------------
    # APPLY TRAINED SCALER
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X_processed
    )

    return X_scaled


# ============================================================
# CONVERT NUMERIC PREDICTIONS TO LABELS
# ============================================================

def convert_predictions_to_labels(
    predictions,
    artifacts
):
    """
    Converts model predictions such as:

        0 -> <=50K
        1 -> >50K

    using inverse_target_mapping saved in preprocessing.pkl.
    """

    inverse_mapping = artifacts[
        "inverse_target_mapping"
    ]

    labels = []

    for prediction in predictions:

        prediction_value = int(
            prediction
        )

        prediction_key = str(
            prediction_value
        )

        if prediction_key not in inverse_mapping:

            raise ValueError(
                f"No label mapping found for "
                f"prediction: {prediction_value}"
            )

        labels.append(
            inverse_mapping[
                prediction_key
            ]
        )

    return labels


# ============================================================
# CONVERT ACTUAL LABELS TO NUMERIC VALUES
# ============================================================

def convert_actual_labels_to_numeric(
    actual_labels,
    artifacts
):
    """
    Converts actual target labels to their numeric values.

    Example:
        <=50K -> 0
        >50K  -> 1
    """

    target_mapping = artifacts[
        "target_mapping"
    ]

    numeric_values = []

    for label in actual_labels:

        label_string = str(
            label
        ).strip()

        if label_string not in target_mapping:

            raise ValueError(
                f"Unknown target label: "
                f"'{label_string}'. "
                f"Expected one of: "
                f"{list(target_mapping.keys())}"
            )

        numeric_values.append(
            int(
                target_mapping[
                    label_string
                ]
            )
        )

    return np.asarray(
        numeric_values,
        dtype=int
    )


# ============================================================
# GET PREDICTION PROBABILITIES
# ============================================================

def get_positive_class_probabilities(
    model,
    X_scaled
):
    """
    Returns probability scores for the positive class.

    These are used to calculate a proper ROC-AUC score.
    """

    if not hasattr(
        model,
        "predict_proba"
    ):

        return None

    probabilities = model.predict_proba(
        X_scaled
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim != 2:

        return None

    if probabilities.shape[1] < 2:

        return None

    # Find the column corresponding to class 1
    if hasattr(model, "classes_"):

        classes = list(
            model.classes_
        )

        if 1 in classes:

            positive_index = classes.index(1)

        else:

            positive_index = 1

    else:

        positive_index = 1

    return probabilities[
        :,
        positive_index
    ]


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_scaled,
    actual_labels,
    artifacts
):
    """
    Calculates:

        Accuracy
        AUC
        Precision
        Recall
        F1
        MCC
        Confusion Matrix
        Classification Report
    """

    # --------------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        X_scaled
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
    ).reshape(-1)

    # --------------------------------------------------------
    # ACTUAL LABELS
    # --------------------------------------------------------

    y_true = convert_actual_labels_to_numeric(
        actual_labels,
        artifacts
    )

    # --------------------------------------------------------
    # VALIDATE DIMENSIONS
    # --------------------------------------------------------

    if len(y_true) != len(y_pred):

        raise ValueError(
            f"Actual target count ({len(y_true)}) "
            f"does not match prediction count "
            f"({len(y_pred)})."
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # AUC USING PREDICTION PROBABILITIES
    # --------------------------------------------------------

    positive_probabilities = (
        get_positive_class_probabilities(
            model,
            X_scaled
        )
    )

    try:

        if positive_probabilities is not None:

            auc = roc_auc_score(
                y_true,
                positive_probabilities
            )

        else:

            auc = roc_auc_score(
                y_true,
                y_pred
            )

    except ValueError:

        auc = float("nan")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "<=50K",
            ">50K"
        ],
        output_dict=True,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred
    }


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "💰 Adult Income Classification"
)

st.markdown(
    """
Predict whether an individual's income belongs to the
**<=50K** or **>50K** category using trained machine
learning classification models.

The application loads trained **PKL models** and applies
the same preprocessing used during model training.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Model Selection"
)

selected_model = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    list(MODEL_CONFIG.keys()),
    key="selected_model"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "### Available Models"
)

st.sidebar.write(
    """
- Random Forest
- Decision Tree
- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors
"""
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Prediction CSV**

The target column is optional.

**Evaluation CSV**

Include the `income` target column to calculate
evaluation metrics, the confusion matrix, and the
classification report.
"""
)


# ============================================================
# MODEL FILE STATUS
# ============================================================

with st.expander(
    "🔍 Model File Status",
    expanded=False
):

    # --------------------------------------------------------
    # CHECK PREPROCESSING FILE
    # --------------------------------------------------------

    if PREPROCESSING_PATH.exists():

        st.success(
            "✓ Preprocessing: "
            f"`{PREPROCESSING_PATH.name}`"
        )

    else:

        st.error(
            "✗ Preprocessing file not found: "
            f"`{PREPROCESSING_PATH}`"
        )

    # --------------------------------------------------------
    # CHECK MODEL FILES
    # --------------------------------------------------------

    for model_name, model_path in (
        MODEL_CONFIG.items()
    ):

        if model_path.exists():

            st.success(
                f"✓ {model_name}: "
                f"`{model_path.name}`"
            )

        else:

            st.error(
                f"✗ {model_name}: "
                f"`{model_path.name}` not found"
            )


# ============================================================
# LOAD PREPROCESSING ARTIFACTS
# ============================================================

try:

    artifacts = load_preprocessing_artifacts()

except Exception as preprocessing_error:

    st.error(
        "Unable to load preprocessing artifacts."
    )

    st.exception(
        preprocessing_error
    )

    st.stop()


# ============================================================
# FILE UPLOAD
# ============================================================

st.header(
    "📂 Upload Input Data"
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

st.caption(
    "For prediction only, upload a CSV containing the "
    "feature columns. To calculate evaluation metrics, "
    "include the target column as well."
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    try:

        input_df = pd.read_csv(
            uploaded_file
        )

    except Exception as file_error:

        st.error(
            "Unable to read the uploaded CSV file: "
            f"{file_error}"
        )

        st.stop()

    # --------------------------------------------------------
    # DISPLAY INPUT DATA
    # --------------------------------------------------------

    st.subheader(
        "📋 Uploaded Data"
    )

    st.dataframe(
        input_df,
        use_container_width=True
    )

    st.write(
        f"Rows: **{input_df.shape[0]}** | "
        f"Columns: **{input_df.shape[1]}**"
    )

    # --------------------------------------------------------
    # CHECK TARGET COLUMN
    # --------------------------------------------------------

    target_available = (
        TARGET_COLUMN in input_df.columns
    )

    if target_available:

        st.success(
            f"Target column `{TARGET_COLUMN}` detected. "
            "Evaluation metrics will be available."
        )

    else:

        st.info(
            f"Target column `{TARGET_COLUMN}` was not found. "
            "The application will perform prediction only."
        )

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    if target_available:

        X_input = input_df.drop(
            columns=[TARGET_COLUMN]
        )

        actual_labels = input_df[
            TARGET_COLUMN
        ]

    else:

        X_input = input_df.copy()

        actual_labels = None

    # --------------------------------------------------------
    # LOAD SELECTED MODEL
    # --------------------------------------------------------

    try:

        model = load_model(
            selected_model
        )

    except Exception as model_error:

        st.error(
            f"Unable to load **{selected_model}**."
        )

        st.exception(
            model_error
        )

        st.stop()

    st.success(
        f"Model loaded successfully: "
        f"**{selected_model}**"
    )

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Income",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # PREPROCESS AND PREDICT
        # ----------------------------------------------------

        try:

            with st.spinner(
                f"Processing data and generating predictions "
                f"using {selected_model}..."
            ):

                X_scaled = preprocess_input(
                    X_input,
                    artifacts
                )

                numeric_predictions = model.predict(
                    X_scaled
                )

                numeric_predictions = np.asarray(
                    numeric_predictions,
                    dtype=int
                ).reshape(-1)

                predictions = (
                    convert_predictions_to_labels(
                        numeric_predictions,
                        artifacts
                    )
                )

        except Exception as prediction_error:

            st.error(
                "An error occurred while generating predictions."
            )

            st.exception(
                prediction_error
            )

            st.stop()

        # ----------------------------------------------------
        # VALIDATE PREDICTION COUNT
        # ----------------------------------------------------

        if len(predictions) != len(X_input):

            st.error(
                f"Prediction count mismatch. "
                f"Received {len(predictions)} predictions "
                f"for {len(X_input)} input rows."
            )

            st.stop()

        # ----------------------------------------------------
        # RESULTS DATAFRAME
        # ----------------------------------------------------

        results_df = input_df.copy()

        results_df[
            "Predicted Income"
        ] = predictions

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        st.subheader(
            f"🎯 Prediction Results - {selected_model}"
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # PREDICTION SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "📊 Prediction Summary"
        )

        prediction_counts = (
            pd.Series(predictions)
            .value_counts()
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Predictions",
                len(predictions)
            )

        with col2:

            st.metric(
                "Unique Income Classes",
                len(prediction_counts)
            )

        # ----------------------------------------------------
        # PREDICTION DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "📈 Predicted Income Distribution"
        )

        distribution_df = (
            prediction_counts
            .rename_axis(
                "Income Class"
            )
            .reset_index(
                name="Count"
            )
        )

        st.dataframe(
            distribution_df,
            use_container_width=True
        )

        st.bar_chart(
            distribution_df.set_index(
                "Income Class"
            )
        )

        # ====================================================
        # EVALUATION SECTION
        # ====================================================

        if target_available:

            st.markdown("---")

            st.header(
                f"📊 Model Evaluation - {selected_model}"
            )

            try:

                evaluation = evaluate_model(
                    model,
                    X_scaled,
                    actual_labels,
                    artifacts
                )

                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                st.subheader(
                    "Evaluation Metrics"
                )

                metric_col1, metric_col2, metric_col3 = (
                    st.columns(3)
                )

                with metric_col1:

                    st.metric(
                        "Accuracy",
                        f"{evaluation['Accuracy']:.4f}"
                    )

                    st.metric(
                        "Precision (Macro)",
                        f"{evaluation['Precision']:.4f}"
                    )

                with metric_col2:

                    st.metric(
                        "AUC",
                        f"{evaluation['AUC']:.4f}"
                    )

                    st.metric(
                        "Recall (Macro)",
                        f"{evaluation['Recall']:.4f}"
                    )

                with metric_col3:

                    st.metric(
                        "F1-Score (Macro)",
                        f"{evaluation['F1']:.4f}"
                    )

                    st.metric(
                        "MCC",
                        f"{evaluation['MCC']:.4f}"
                    )

                # --------------------------------------------
                # METRICS TABLE
                # --------------------------------------------

                metrics_df = pd.DataFrame({

                    "Metric": [
                        "Accuracy",
                        "AUC",
                        "Precision",
                        "Recall",
                        "F1-Score",
                        "MCC"
                    ],

                    "Score": [
                        evaluation["Accuracy"],
                        evaluation["AUC"],
                        evaluation["Precision"],
                        evaluation["Recall"],
                        evaluation["F1"],
                        evaluation["MCC"]
                    ]
                })

                st.dataframe(
                    metrics_df.style.format(
                        {"Score": "{:.4f}"}
                    ),
                    use_container_width=True
                )

                # --------------------------------------------
                # CONFUSION MATRIX
                # --------------------------------------------

                st.subheader(
                    "Confusion Matrix"
                )

                cm = evaluation[
                    "confusion_matrix"
                ]

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        "Actual <=50K",
                        "Actual >50K"
                    ],
                    columns=[
                        "Predicted <=50K",
                        "Predicted >50K"
                    ]
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True
                )

                st.bar_chart(
                    cm_df
                )

                # --------------------------------------------
                # CLASSIFICATION REPORT
                # --------------------------------------------

                st.subheader(
                    "Classification Report"
                )

                report = evaluation[
                    "classification_report"
                ]

                report_rows = []

                for class_name in [
                    "<=50K",
                    ">50K"
                ]:

                    if class_name in report:

                        report_rows.append({

                            "Class": class_name,

                            "Precision": report[
                                class_name
                            ]["precision"],

                            "Recall": report[
                                class_name
                            ]["recall"],

                            "F1-Score": report[
                                class_name
                            ]["f1-score"],

                            "Support": report[
                                class_name
                            ]["support"]
                        })

                report_df = pd.DataFrame(
                    report_rows
                )

                st.dataframe(
                    report_df.style.format({
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "Support": "{:.0f}"
                    }),
                    use_container_width=True
                )

            except Exception as evaluation_error:

                st.error(
                    "Unable to calculate evaluation metrics."
                )

                st.exception(
                    evaluation_error
                )

        # ====================================================
        # NO TARGET COLUMN
        # ====================================================

        else:

            st.info(
                """
                **Evaluation metrics are not displayed because
                the uploaded CSV does not contain the target
                column.**

                Upload a CSV containing the actual `income`
                label to calculate Accuracy, AUC, Precision,
                Recall, F1-Score, MCC, Confusion Matrix, and
                Classification Report.
                """
            )

        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.markdown("---")

        st.subheader(
            "💾 Download Results"
        )

        csv_output = results_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_output,
            file_name=(
                "adult_income_predictions.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# NO FILE UPLOADED
# ============================================================

else:

    st.info(
        """
        👆 Upload a CSV file to begin.

        **Prediction mode:** the `income` target column
        is optional.

        **Evaluation mode:** include the `income` target
        column to display Accuracy, AUC, Precision, Recall,
        F1-Score, MCC, Confusion Matrix, and
        Classification Report.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Adult Income Classification | "
    "Machine Learning Model Deployment | "
    "2025AC05855"
)
