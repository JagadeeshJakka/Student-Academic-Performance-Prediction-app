import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

# --- DATA & MODEL LOADING ---
@st.cache_data
def get_data_and_model():
    # Load dataset directly from the local file
    df = pd.read_csv('student_performance_dataset.csv')
    
    # Preprocessing (Matching your notebook logic)
    df_clean = df.copy()
    le = LabelEncoder()
    
    # Categorical columns to encode
    cat_cols = ['internet_access', 'part_time_job', 'extra_classes', 'pass_fail']
    for col in cat_cols:
        df_clean[col] = le.fit_transform(df_clean[col])
    
    # Features and Target
    X = df_clean.drop(['student_id', 'final_exam_score', 'pass_fail'], axis=1)
    y = df_clean['pass_fail']
    
    # Train the Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return df, model, X.columns, le

# Initialize data
df, model, feature_cols, label_encoder = get_data_and_model()

# --- APP UI ---
st.title("🎓 Student Performance: Automated Prediction")
st.info("The dataset has been loaded automatically. Select a Student ID to begin.")

# 1. Searchable Selection
st.subheader("Step 1: Select Student")
student_list = df['student_id'].tolist()
selected_id = st.selectbox("Search for Student ID (e.g. S1001):", student_list)

# 2. Display Student Profile
if selected_id:
    student_info = df[df['student_id'] == selected_id].iloc[0]
    
    st.markdown("### 📋 Student Profile Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Age", f"{student_info['age']}")
        st.metric("Study Hours", f"{student_info['study_hours_per_day']} /day")
    
    with col2:
        st.metric("Attendance", f"{student_info['attendance_percent']}%")
        st.metric("Assignments", f"{student_info['assignments_score']}/100")
        
    with col3:
        st.metric("Extra Classes", student_info['extra_classes'])
        st.metric("Internet", student_info['internet_access'])

    st.divider()

    # 3. Automated Prediction
    if st.button("Generate Prediction Result"):
        # Prepare data for model
        input_data = df[df['student_id'] == selected_id].copy()
        
        # Local encoding for the prediction row
        for col in ['internet_access', 'part_time_job', 'extra_classes']:
            # We map Yes/No to match the training (Yes=1, No=0 usually)
            input_data[col] = input_data[col].map({'Yes': 1, 'No': 0})
            
        features = input_data[feature_cols]
        prediction = model.predict(features)
        prob = model.predict_proba(features)

        # 4. Display Results
        st.subheader("Result Analysis")
        if prediction[0] == 1: # Assuming 'Pass' was encoded as 1
            st.success(f"### Prediction: PASS")
        else:
            st.error(f"### Prediction: FAIL")
            
        st.write(f"**Model Confidence:** {np.max(prob)*100:.2f}%")
        st.progress(float(np.max(prob)))
