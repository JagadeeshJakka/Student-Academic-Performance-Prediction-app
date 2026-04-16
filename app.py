import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Student Academic Performance Prediction",
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
def load_default_dataset():
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
        "student_id",
        "age",
        "study_hours_per_day",
        "attendance_percent",
        "assignments_score",
        "previous_marks",
        "internet_access",
        "part_time_job",
        "extra_classes",
        "final_exam_score",
    ]


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
                "median": float(series.median()),
            }
        else:
            values = [str(v) for v in series.astype(str).unique().tolist()]
            mode_val = str(series.astype(str).mode().iloc[0]) if not series.mode().empty else values[0]
            profile[col] = {
                "type": "categorical",
                "values": values,
                "mode": mode_val,
            }

    return profile


def encoder_classes_for_column(label_encoders, col_name):
    if isinstance(label_encoders, dict) and col_name in label_encoders:
        encoder = label_encoders[col_name]
        if hasattr(encoder, "classes_"):
            return [str(x) for x in encoder.classes_]
    return None


def get_prefill_row(df):
    if df is None or df.empty:
        return {}

    st.markdown("### Choose Student Record")
    top_cols = st.columns([2, 2, 1])

    selected_index = 0
    with top_cols[0]:
        if "student_id" in df.columns:
            student_options = df["student_id"].astype(str).tolist()
            selected_student_id = st.selectbox(
                "Select Student ID from Dataset",
                student_options,
                index=0,
                help="Pick an existing student to auto-fill the form."
            )
            matches = df.index[df["student_id"].astype(str) == selected_student_id].tolist()
            if matches:
                selected_index = matches[0]
        else:
            selected_index = 0

    with top_cols[1]:
        row_number = st.number_input(
            "Or choose row number",
            min_value=0,
            max_value=max(len(df) - 1, 0),
            value=int(selected_index),
            step=1,
            help="This will also auto-fill the form using the selected row."
        )
        selected_index = int(row_number)

    with top_cols[2]:
        use_defaults = st.toggle("Use dataset row", value=True)

    if use_defaults:
        return df.iloc[selected_index].to_dict()
    return {}


def build_input_form(feature_columns, label_encoders, dataset_profile, prefill_row):
    st.markdown("### Student Input Form")
    st.caption("The fields below are built from the uploaded CSV values.")

    user_values = {}
    cols = st.columns(2)

    for i, col_name in enumerate(feature_columns):
        with cols[i % 2]:
            label = col_name.replace("_", " ").title()
            lower = col_name.lower()
            profile = dataset_profile.get(col_name, {})
            encoder_classes = encoder_classes_for_column(label_encoders, col_name)
            prefill_value = prefill_row.get(col_name, None)

            if "student_id" in lower:
                options = profile.get("values", [])
                if prefill_value is not None:
                    prefill_value = str(prefill_value)
                if not options and prefill_value is not None:
                    options = [prefill_value]
                default_index = options.index(prefill_value) if prefill_value in options else 0
                user_values[col_name] = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    help="Student IDs are loaded from the dataset.",
                    key=f"input_{col_name}",
                )
                continue

            if encoder_classes:
                options = encoder_classes
                if prefill_value is not None:
                    prefill_value = str(prefill_value)
                elif profile.get("mode") in options:
                    prefill_value = profile.get("mode")
                else:
                    prefill_value = options[0]

                default_index = options.index(prefill_value) if prefill_value in options else 0
                user_values[col_name] = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    key=f"input_{col_name}",
                )
                continue

            if profile.get("type") == "categorical":
                options = profile.get("values", [])
                if prefill_value is not None:
                    prefill_value = str(prefill_value)
                else:
                    prefill_value = profile.get("mode", options[0] if options else "")
                default_index = options.index(prefill_value) if prefill_value in options else 0
                user_values[col_name] = st.selectbox(
                    label,
                    options=options,
                    index=default_index,
                    key=f"input_{col_name}",
                )
            else:
                min_val = float(profile.get("min", 0.0))
                max_val = float(profile.get("max", max(100.0, min_val + 10)))
                default_val = float(prefill_value) if prefill_value is not None else float(profile.get("median", profile.get("mean", min_val)))

                if lower == "age":
                    user_values[col_name] = st.slider(
                        label,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int(round(default_val)),
                        step=1,
                        key=f"input_{col_name}",
                    )
                elif lower in {"attendance_percent", "assignments_score", "previous_marks", "final_exam_score"}:
                    user_values[col_name] = st.slider(
                        label,
                        min_value=int(min_val),
                        max_value=int(max_val),
                        value=int(round(default_val)),
                        step=1,
                        key=f"input_{col_name}",
                    )
                else:
                    user_values[col_name] = st.slider(
                        label,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default_val),
                        step=0.1,
                        key=f"input_{col_name}",
                    )

    return user_values


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


def pretty_metric_cards(df):
    if df is None or df.empty:
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Records", f"{df.shape[0]}")
    with c2:
        st.metric("Features", f"{df.shape[1]}")
    with c3:
        if "pass_fail" in df.columns:
            pass_rate = (df["pass_fail"].astype(str).str.lower() == "pass").mean() * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        else:
            st.metric("Pass Rate", "N/A")
    with c4:
        if "study_hours_per_day" in df.columns:
            st.metric("Avg Study Hours", f"{df['study_hours_per_day'].mean():.1f}")
        else:
            st.metric("Avg Study Hours", "N/A")


def main():
    st.title("🎓 Student Academic Performance Prediction")
    st.write(
        "Upload your CSV dataset and use a real student record to auto-fill the prediction form. "
        "All dropdowns and input ranges are built from the CSV values."
    )

    try:
        rf_model, lr_model, scaler, label_encoders, metadata = load_artifacts()
    except Exception as e:
        st.error(str(e))
        st.stop()

    feature_columns = safe_feature_columns(metadata)
    target_column = metadata.get("target_column", "pass_fail") if isinstance(metadata, dict) else "pass_fail"
    target_encoder = label_encoders.get(target_column) if isinstance(label_encoders, dict) else None

    default_df, default_name = load_default_dataset()

    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox(
            "Choose Model",
            ["Random Forest", "Logistic Regression"],
            index=0,
        )

        uploaded_file = st.file_uploader(
            "Upload CSV Dataset",
            type=["csv"],
            help="Upload a CSV to refresh all form values from that dataset."
        )

    if uploaded_file is not None:
        try:
            dataset_df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
        except Exception as e:
            st.error(f"Could not read uploaded dataset: {e}")
            st.stop()
    else:
        dataset_df = default_df
        dataset_name = default_name

    if dataset_df is None:
        st.warning("Please upload a CSV dataset or keep one beside app.py.")
        st.stop()

    dataset_profile = get_dataset_profile(dataset_df, feature_columns)

    with st.sidebar:
        st.success(f"Dataset Loaded: {dataset_name}")
        st.write(f"Rows: {dataset_df.shape[0]}")
        st.write(f"Columns: {dataset_df.shape[1]}")

        with st.expander("Model Input Columns"):
            st.write(feature_columns)

    st.markdown("## Dataset Overview")
    pretty_metric_cards(dataset_df)

    missing_columns = [col for col in feature_columns if col not in dataset_df.columns]
    extra_columns = [col for col in dataset_df.columns if col not in feature_columns and col != target_column]

    if missing_columns:
        st.warning(f"Missing columns in dataset: {missing_columns}")
    if extra_columns:
        st.info(f"Extra dataset columns not used by the model: {extra_columns}")

    prefill_row = get_prefill_row(dataset_df)

    with st.expander("Preview Selected Dataset Rows", expanded=False):
        preview_columns = [col for col in dataset_df.columns if col in feature_columns or col == target_column]
        st.dataframe(dataset_df[preview_columns].head(10), use_container_width=True)

    user_values = build_input_form(feature_columns, label_encoders, dataset_profile, prefill_row)

    if st.button("Predict Result", use_container_width=True):
        try:
            input_df = pd.DataFrame([user_values])

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

            st.markdown("## Prediction Output")
            out1, out2 = st.columns([1, 1])

            with out1:
                if prediction_label.strip().lower() == "pass":
                    st.success(f"Predicted Result: {prediction_label}")
                else:
                    st.error(f"Predicted Result: {prediction_label}")

                if probability is not None:
                    st.info(f"Probability of Pass: {probability:.2f}%")

            with out2:
                st.write("Selected Model")
                st.code(model_choice)

            st.markdown("### Processed Input Used by the Model")
            st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.caption("UI updated from the CSV dataset: dropdowns, ranges, defaults, and student record auto-fill.")


if __name__ == "__main__":
    main()
