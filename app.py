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

    missing = [str(path.name) for path in files.values() if not path.exists()]
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


@st.cache_data
def load_dataset():
    candidates = [
        BASE_DIR / "student_performance_dataset.csv",
        BASE_DIR / "dataset.csv",
        BASE_DIR / "data.csv",
        BASE_DIR / "students.csv",
    ]

    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df, path.name
            except Exception:
                pass

    return None, None


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
            return [str(x) for x in encoder.classes_]
    return None


def get_dataset_profile(df, feature_columns):
    profile = {}

    if df is None:
        return profile

    for col in feature_columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            profile[col] = {
                "type": "numeric",
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
            }
        else:
            unique_values = [str(v) for v in series.astype(str).unique().tolist()]
            profile[col] = {
                "type": "categorical",
                "values": unique_values,
                "mode": str(series.astype(str).mode().iloc[0]) if not series.mode().empty else unique_values[0],
            }

    return profile


def build_input_form(feature_columns, label_encoders, dataset_profile):
    st.subheader("Enter Student Details")

    user_values = {}
    cols = st.columns(2)

    for i, col_name in enumerate(feature_columns):
        with cols[i % 2]:
            lower = col_name.lower()
            profile = dataset_profile.get(col_name, {})
            encoder_classes = encoder_classes_for_column(label_encoders, col_name)

            if encoder_classes:
                default_value = encoder_classes[0]
                if profile.get("type") == "categorical" and profile.get("mode") in encoder_classes:
                    default_value = profile["mode"]

                default_index = encoder_classes.index(default_value)
                user_values[col_name] = st.selectbox(
                    col_name.replace("_", " ").title(),
                    encoder_classes,
                    index=default_index,
                )
                continue

            if profile.get("type") == "categorical":
                options = profile.get("values", ["Yes", "No"])
                default_value = profile.get("mode", options[0])
                default_index = options.index(default_value) if default_value in options else 0
                user_values[col_name] = st.selectbox(
                    col_name.replace("_", " ").title(),
                    options,
                    index=default_index,
                )
            elif "student_id" in lower:
                default_value = "S1001"
                user_values[col_name] = st.text_input(
                    col_name.replace("_", " ").title(),
                    value=default_value,
                    help="Use the same format used in training.",
                )
            else:
                min_val = float(profile.get("min", 0.0))
                max_val = float(profile.get("max", max(100.0, min_val + 10)))
                mean_val = float(profile.get("mean", (min_val + max_val) / 2))

                step = 1 if lower == "age" else 0.1

                if lower == "age":
                    user_values[col_name] = st.number_input(
                        col_name.replace("_", " ").title(),
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int(round(mean_val)),
                        step=1,
                    )
                else:
                    user_values[col_name] = st.number_input(
                        col_name.replace("_", " ").title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=step,
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
            allowed = [str(x) for x in encoder.classes_]
            if value not in allowed:
                raise ValueError(
                    f"Value '{value}' for column '{col}' was not seen during training. "
                    f"Allowed values: {allowed}"
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
        "using your trained machine learning models."
    )

    try:
        rf_model, lr_model, scaler, label_encoders, metadata = load_artifacts()
    except Exception as e:
        st.error(str(e))
        st.stop()

    dataset_df, dataset_name = load_dataset()
    feature_columns = safe_feature_columns(metadata)
    dataset_profile = get_dataset_profile(dataset_df, feature_columns)

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
        st.success("Keep all .pkl files in the same folder as app.py")

        if dataset_df is not None:
            st.success(f"Dataset loaded: {dataset_name}")
            st.write(f"Rows: {dataset_df.shape[0]}")
            st.write(f"Columns: {dataset_df.shape[1]}")
        else:
            st.warning(
                "Dataset file not found. Add one of these in the same folder: "
                "`student_performance_dataset.csv`, `dataset.csv`, `data.csv`, or `students.csv`"
            )

        with st.expander("Show expected input columns"):
            st.write(feature_columns)

    user_values = build_input_form(feature_columns, label_encoders, dataset_profile)

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

    if dataset_df is not None:
        st.markdown("---")
        st.subheader("Dataset Preview")
        st.dataframe(dataset_df.head(), use_container_width=True)

    st.markdown("---")
    st.caption("Built with Streamlit. Input ranges and dropdowns are pulled from the dataset when available.")


if __name__ == "__main__":
    main()
