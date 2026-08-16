# ============================================================
# STREAMLIT APP - ADULT INCOME CLASSIFICATION
# ============================================================

from pathlib import Path
import importlib.util

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
    "Random Forest": MODEL_DIR / "random_forest_model.py",
    "Decision Tree": MODEL_DIR / "decision_tree_model.py",
    "Logistic Regression": MODEL_DIR / "logistic_regression_model.py",
    "Gaussian Naive Bayes": MODEL_DIR / "gaussian_nb_model.py",
    "K-Nearest Neighbors": MODEL_DIR / "k_nearest_neighbors_model.py",
}


# ============================================================
# TARGET CONFIGURATION
# ============================================================
# This must match the target_mapping used during training.
#
# Example:
# "<=50K" -> 0
# ">50K"  -> 1
#
# The exported model's predict_labels() is preferred for
# converting numeric predictions back to their original labels.

TARGET_COLUMN = "income"


# ============================================================
# LOAD MODEL MODULE
# ============================================================

@st.cache_resource
def load_model_module(model_name):
    """
    Dynamically loads one of the exported model Python modules.

    Required:
        predict(data)

    Preferred:
        predict_labels(data)
    """

    model_path = MODEL_CONFIG[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    module_name = (
        "model_"
        + model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        model_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load model module:\n{model_path}"
        )

    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)

    except Exception as error:
        raise RuntimeError(
            f"Error while loading {model_name}: {error}"
        ) from error

    if not hasattr(module, "predict"):
        raise AttributeError(
            f"{model_name} module does not contain predict()."
        )

    return module


# ============================================================
# GET NUMERIC PREDICTIONS
# ============================================================

def get_numeric_predictions(model_module, data):
    """
    Returns the raw numeric predictions from the generated
    model module.

    These are required for evaluation metrics such as AUC.
    """

    predictions = model_module.predict(data)

    if not isinstance(predictions, list):
        predictions = [predictions]

    return predictions


# ============================================================
# GET DISPLAY LABEL PREDICTIONS
# ============================================================

def get_label_predictions(model_module, data):
    """
    Returns predictions converted to their original target
    labels using the model module's INVERSE_TARGET_MAPPING.

    This avoids hard-coding:
        0 -> <=50K
        1 -> >50K
    """

    if hasattr(model_module, "predict_labels"):

        predictions = model_module.predict_labels(data)

    else:

        predictions = model_module.predict(data)

    if not isinstance(predictions, list):
        predictions = [predictions]

    return predictions


# ============================================================
# CONVERT ACTUAL LABELS TO NUMERIC VALUES
# ============================================================

def convert_actual_labels_to_numeric(
    model_module,
    actual_labels
):
    """
    Converts actual target labels into numeric values using
    the exported model's INVERSE_TARGET_MAPPING.

    Example:
        <=50K -> 0
        >50K  -> 1
    """

    if not hasattr(
        model_module,
        "INVERSE_TARGET_MAPPING"
    ):
        raise AttributeError(
            "The exported model module does not contain "
            "INVERSE_TARGET_MAPPING."
        )

    inverse_mapping = (
        model_module.INVERSE_TARGET_MAPPING
    )

    # INVERSE_TARGET_MAPPING has:
    # "0" -> "<=50K"
    # "1" -> ">50K"

    label_to_numeric = {
        str(label): int(code)
        for code, label in inverse_mapping.items()
    }

    numeric_values = []

    for label in actual_labels:

        label_string = str(label).strip()

        if label_string not in label_to_numeric:
            raise ValueError(
                f"Unknown target label '{label_string}'. "
                f"Expected one of: "
                f"{list(label_to_numeric.keys())}"
            )

        numeric_values.append(
            label_to_numeric[label_string]
        )

    return np.asarray(
        numeric_values,
        dtype=int
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_module,
    X,
    actual_labels
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
    # Numeric predictions
    # --------------------------------------------------------

    numeric_predictions = get_numeric_predictions(
        model_module,
        X
    )

    y_pred = np.asarray(
        numeric_predictions,
        dtype=int
    ).reshape(-1)

    # --------------------------------------------------------
    # Actual target values
    # --------------------------------------------------------

    y_true = convert_actual_labels_to_numeric(
        model_module,
        actual_labels
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Actual target count ({len(y_true)}) "
            f"does not match prediction count "
            f"({len(y_pred)})."
        )

    # --------------------------------------------------------
    # Evaluation metrics
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
    # AUC
    # --------------------------------------------------------
    #
    # For this binary deployment implementation, numeric
    # predictions are used for AUC if probability scores are
    # not available from the exported module.
    #
    # AUC calculated this way is based on the predicted class
    # labels, not probability estimates.
    #
    # The notebook's original AUC values were calculated from
    # model.predict_proba(), so they should be used as the
    # official training/test evaluation results.
    # --------------------------------------------------------

    try:

        auc = roc_auc_score(
            y_true,
            y_pred
        )

    except ValueError:

        auc = float("nan")

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # Classification report
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

st.title("💰 Adult Income Classification")

st.markdown(
    """
Predict whether an individual's income belongs to the
**<=50K** or **>50K** category using trained machine
learning classification models.

The application uses the same preprocessing logic and
exported model parameters used during training.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    list(MODEL_CONFIG.keys())
)

st.sidebar.markdown("---")

st.sidebar.write("### Available Models")

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

Target column is optional.

**Evaluation CSV**

Include the target column to calculate
evaluation metrics and the confusion matrix.
"""
)


# ============================================================
# MODEL FILE STATUS
# ============================================================

with st.expander(
    "🔍 Model File Status",
    expanded=False
):

    for model_name, model_path in MODEL_CONFIG.items():

        if model_path.exists():

            st.success(
                f"✓ {model_name}: `{model_path.name}`"
            )

        else:

            st.error(
                f"✗ {model_name}: `{model_path}` not found"
            )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Input Data")

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
# PROCESS FILE
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
            f"Unable to read the uploaded CSV file: "
            f"{file_error}"
        )

        st.stop()

    # --------------------------------------------------------
    # DISPLAY INPUT DATA
    # --------------------------------------------------------

    st.subheader("📋 Uploaded Data")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    st.write(
        f"Rows: **{input_df.shape[0]}** | "
        f"Columns: **{input_df.shape[1]}**"
    )

    # --------------------------------------------------------
    # CHECK FOR TARGET COLUMN
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
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model_module = load_model_module(
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
        f"Model loaded successfully: **{selected_model}**"
    )

    # --------------------------------------------------------
    # REMOVE TARGET COLUMN FOR PREDICTION
    # --------------------------------------------------------

    if target_available:

        X_input = input_df.drop(
            columns=[TARGET_COLUMN]
        )

        actual_labels = input_df[
            TARGET_COLUMN
        ]

    else:

        X_input = input_df
        actual_labels = None

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Income",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # GENERATE LABEL PREDICTIONS
        # ----------------------------------------------------

        try:

            with st.spinner(
                f"Generating predictions using "
                f"{selected_model}..."
            ):

                predictions = get_label_predictions(
                    model_module,
                    X_input
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
            "🎯 Prediction Results"
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
            .rename_axis("Income Class")
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
                "📊 Model Evaluation"
            )

            try:

                evaluation = evaluate_model(
                    model_module,
                    X_input,
                    actual_labels
                )

                # ------------------------------------------------
                # METRICS
                # ------------------------------------------------

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

                # ------------------------------------------------
                # METRICS TABLE
                # ------------------------------------------------

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

                # ------------------------------------------------
                # CONFUSION MATRIX
                # ------------------------------------------------

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

                # ------------------------------------------------
                # CLASSIFICATION REPORT
                # ------------------------------------------------

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

        else:

            st.info(
                """
                **Evaluation metrics are not displayed because
                the uploaded CSV does not contain the target
                column.**

                Upload a CSV containing the actual income label
                to calculate Accuracy, AUC, Precision, Recall,
                F1-Score, MCC, Confusion Matrix, and Classification
                Report.
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

        **Prediction mode:** target column is optional.

        **Evaluation mode:** include the target column
        to display Accuracy, AUC, Precision, Recall,
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
