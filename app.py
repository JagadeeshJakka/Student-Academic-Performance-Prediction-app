import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    files = {
        "random_forest_model": BASE_DIR / "random_forest_model.pkl",
        "logistic_regression_model": BASE_DIR / "logistic_regression_model.pkl",
        "scaler": BASE_DIR / "scaler.pkl",
        "label_encoders": BASE_DIR / "label_encoders.pkl",
        "metadata": BASE_DIR / "metadata.pkl",
    }

    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files: " + ", ".join(missing) +
            ". Upload all .pkl files in the same folder as app.py."
        )

    with open(files["random_forest_model"], "rb") as f:
        rf_model = pickle.load(f)

    with open(files["logistic_regression_model"], "rb") as f:
        lr_model = pickle.load(f)

    with open(files["scaler"], "rb") as f:
        scaler = pickle.load(f)

    with open(files["label_encoders"], "rb") as f:
        label_encoders = pickle.load(f)

    with open(files["metadata"], "rb") as f:
        metadata = pickle.load(f)

    return rf_model, lr_model, scaler, label_encoders, metadata


def safe_feature_columns(metadata):
    if isinstance(metadata, dict) and "feature_columns" in metadata:
        cols = metadata["feature_columns"]
        if isinstance(cols, list) and cols:
            return cols

    return [
        "age",
        "study_hours_per_day",
        "attendance_percent",
        "assignments_score",
        "previous_marks",
        "internet_access",
        "part_time_job",
        "extra_classes",
    ]


def encoder_classes_for_column(label_encoders, col_name):
    if isinstance(label_encoders, dict) and col_name in label_encoders:
        encoder = label_encoders[col_name]
        if hasattr(encoder, "classes_"):
            return list(encoder.classes_)
    return None


def build_input_form(feature_columns, label_encoders):
    st.subheader("Enter Student Details")

    common_defaults = {
        "age": 21,
        "study_hours_per_day": 3.0,
        "attendance_percent": 80.0,
        "assignments_score": 70.0,
        "previous_marks": 68.0,
        "final_exam_score": 65.0,
        "study_efficiency": 240.0,
        "engagement_score": 75.0,
        "performance_gap": 0.0,
        "internet_access": "Yes",
        "part_time_job": "No",
        "extra_classes": "No",
        "student_id": "S1001",
    }

    user_values = {}

    cols = st.columns(2)

    for i, col_name in enumerate(feature_columns):
        with cols[i % 2]:
            lower = col_name.lower()
            classes = encoder_classes_for_column(label_encoders, col_name)

            if classes is not None and len(classes) > 0:
                default_value = common_defaults.get(col_name, common_defaults.get(lower, classes[0]))
                default_index = classes.index(default_value) if default_value in classes else 0
                user_values[col_name] = st.selectbox(
                    col_name.replace("_", " ").title(),
                    classes,
                    index=default_index,
                )
                continue

            if lower in {"internet_access", "part_time_job", "extra_classes"}:
                options = ["Yes", "No"]
                default_value = common_defaults.get(lower, "Yes")
                default_index = options.index(default_value) if default_value in options else 0
                user_values[col_name] = st.selectbox(
                    col_name.replace("_", " ").title(),
                    options,
                    index=default_index,
                )
            elif "student_id" in lower:
                user_values[col_name] = st.text_input(
                    col_name.replace("_", " ").title(),
                    value=str(common_defaults.get("student_id", "S1001")),
                    help="Use the same format used in training. If your model was trained with encoded student IDs, unseen IDs may not work well.",
                )
            elif lower in {"age"}:
                user_values[col_name] = st.number_input(
                    col_name.replace("_", " ").title(),
                    min_value=10,
                    max_value=100,
                    value=int(common_defaults.get(lower, 20)),
                    step=1,
                )
            else:
                user_values[col_name] = st.number_input(
                    col_name.replace("_", " ").title(),
                    value=float(common_defaults.get(lower, 0.0)),
                    step=0.1,
                )

    return user_values


def add_engineered_features_if_needed(input_data):
    data = input_data.copy()

    if (
        "study_efficiency" in data.columns
        and "study_hours_per_day" in data.columns
        and "attendance_percent" in data.columns
    ):
        data["study_efficiency"] = data["study_hours_per_day"] * data["attendance_percent"]

    if (
        "engagement_score" in data.columns
        and "attendance_percent" in data.columns
        and "assignments_score" in data.columns
        and "extra_classes" in data.columns
    ):
        extra_val = data["extra_classes"].iloc[0]
        if isinstance(extra_val, str):
            extra_bonus = 10 if extra_val.strip().lower() == "yes" else 0
        else:
            extra_bonus = 10 if float(extra_val) == 1 else 0
        data["engagement_score"] = (
            (data["attendance_percent"] + data["assignments_score"]) / 2
        ) + extra_bonus

    if (
        "performance_gap" in data.columns
        and "previous_marks" in data.columns
        and "assignments_score" in data.columns
    ):
        data["performance_gap"] = data["previous_marks"] - data["assignments_score"]

    return data


def encode_categorical_columns(input_df, label_encoders):
    df = input_df.copy()

    if not isinstance(label_encoders, dict):
        return df

    for col, encoder in label_encoders.items():
        if col not in df.columns:
            continue

        if hasattr(encoder, "classes_") and df[col].dtype == "object":
            value = str(df[col].iloc[0])
            if value not in encoder.classes_:
                raise ValueError(
                    f"Value '{value}' for column '{col}' was not seen during training. "
                    f"Allowed values: {list(encoder.classes_)}"
                )
            df[col] = encoder.transform(df[col].astype(str))

    return df


def map_prediction_label(pred, target_encoder=None):
    try:
        pred_value = int(pred[0])
    except Exception:
        pred_value = pred[0]

    if target_encoder is not None and hasattr(target_encoder, "inverse_transform"):
        try:
            return str(target_encoder.inverse_transform([pred_value])[0])
        except Exception:
            pass

    if str(pred_value) in {"1", "1.0"}:
        return "Pass"
    if str(pred_value) in {"0", "0.0"}:
        return "Fail"

    return str(pred_value)


def model_probability(model, X_scaled):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        if len(proba) == 2:
            return float(proba[1]) * 100
    return None


def main():
    st.title("🎓 Student Academic Performance Prediction")
    st.write(
        "This app predicts whether a student is likely to **Pass** or **Fail** "
        "based on the trained machine learning models."
    )

    try:
        rf_model, lr_model, scaler, label_encoders, metadata = load_artifacts()
    except Exception as e:
        st.error(str(e))
        st.stop()

    feature_columns = safe_feature_columns(metadata)
    target_column = metadata.get("target_column", "pass_fail") if isinstance(metadata, dict) else "pass_fail"
    target_encoder = label_encoders.get(target_column) if isinstance(label_encoders, dict) else None

    with st.sidebar:
        st.header("Model Settings")
        model_choice = st.selectbox(
            "Choose model",
            ["Random Forest", "Logistic Regression"],
            index=0,
        )

        st.markdown("### Files")
        st.success("All .pkl files should be in the same folder as app.py")

        if "student_id" in feature_columns or "final_exam_score" in feature_columns:
            st.warning(
                "Your saved model appears to use `student_id` or `final_exam_score`. "
                "That can reduce real-world usefulness or cause leakage. "
                "Retraining without those columns is recommended."
            )

        with st.expander("Show expected input columns"):
            st.write(feature_columns)

    user_values = build_input_form(feature_columns, label_encoders)

    if st.button("Predict Result", use_container_width=True):
        try:
            input_df = pd.DataFrame([user_values])
            input_df = add_engineered_features_if_needed(input_df)

            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[feature_columns]
            input_df = encode_categorical_columns(input_df, label_encoders)

            X_scaled = scaler.transform(input_df)

            model = rf_model if model_choice == "Random Forest" else lr_model
            prediction = model.predict(X_scaled)
            prediction_label = map_prediction_label(prediction, target_encoder=target_encoder)
            probability = model_probability(model, X_scaled)

            st.subheader("Prediction Output")
            if prediction_label.strip().lower() == "pass":
                st.success(f"Predicted Result: {prediction_label}")
            else:
                st.error(f"Predicted Result: {prediction_label}")

            if probability is not None:
                st.info(f"Probability of Pass: {probability:.2f}%")

            st.subheader("Processed Input Used by the Model")
            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.caption(
        "Built with Streamlit. Make sure your model artifacts were saved from the same training pipeline."
    )


if __name__ == "__main__":
    main()
