import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocess import encode_features

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="IT Salary Prediction", layout="centered")
st.title("IT SALARY PREDICTION ANALYSIS")
st.markdown("*Using Linear Regression Model*")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('notebook/model.pkl')
        encoder = joblib.load('notebook/encoder.pkl')
        scaler = joblib.load('notebook/scaler.pkl') 
        return model, encoder, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

model, encoder, scaler = load_model()

# --- RESET SESSION ---
if st.session_state.get("reset_flag"):
    default_values = {
        "work_year_input": 2024,
        "remote_ratio_select": "50",
        "experience_level_select": "Senior (SE)",
        "employment_type_select": "Full-time (FT)",
        "company_size_select": "Medium (M)",
        "company_location_select": "United States (US)",
        "employee_residence_select": "United States (US)",
        "job_title_select": "Data Engineer"
    }
    for key, value in default_values.items():
        st.session_state[key] = value
    st.session_state["reset_flag"] = False

# --- FORM NHẬP LIỆU ---
with st.form(key="salary_prediction_form"):
    st.subheader("Enter Prediction Information")

    col1, col2 = st.columns(2)

    with col1:
        work_year = st.number_input(
            "Working Year*", min_value=2020, max_value=2025,
            value=st.session_state.get("work_year_input", 2024),
            key="work_year_input"
        )
        remote_ratio = st.selectbox(
            "Remote Work Ratio*", options=["0", "50", "100"],
            index=["0", "50", "100"].index(st.session_state.get("remote_ratio_select", "50")),
            key="remote_ratio_select"
        )

    with col2:
        experience_options = {
            "Fresher (EN)": "EN", "Junior (MI)": "MI",
            "Senior (SE)": "SE", "Executive (EX)": "EX"
        }
        employment_options = {
            "Full-time (FT)": "FT", "Part-time (PT)": "PT",
            "Contract (CT)": "CT", "Freelance (FL)": "FL"
        }

        experience_label = st.selectbox(
            "Experience Level*", options=list(experience_options.keys()),
            index=list(experience_options.keys()).index(
                st.session_state.get("experience_level_select", "Senior (SE)")),
            key="experience_level_select"
        )
        experience_level = experience_options[experience_label]

        employment_label = st.selectbox(
            "Employment Type*", options=list(employment_options.keys()),
            index=list(employment_options.keys()).index(
                st.session_state.get("employment_type_select", "Full-time (FT)")),
            key="employment_type_select"
        )
        employment_type = employment_options[employment_label]

    company_size_options = {
        "Small (S)": "S", "Medium (M)": "M", "Large (L)": "L"
    }
    company_size_label = st.selectbox(
        "Company Size*", options=list(company_size_options.keys()),
        index=list(company_size_options.keys()).index(
            st.session_state.get("company_size_select", "Medium (M)")),
        key="company_size_select"
    )
    company_size = company_size_options[company_size_label]

    country_options = {
        "United States (US)": "US", "United Kingdom (GB)": "GB",
        "India (IN)": "IN", "Canada (CA)": "CA", "Germany (DE)": "DE",
        "France (FR)": "FR", "Vietnam (VN)": "VN", "Japan (JP)": "JP", "Australia (AU)": "AU"
    }

    company_location_label = st.selectbox(
        "Company Location*", options=list(country_options.keys()),
        index=list(country_options.keys()).index(
            st.session_state.get("company_location_select", "United States (US)")),
        key="company_location_select"
    )
    company_location = country_options[company_location_label]

    employee_residence_label = st.selectbox(
        "Employee Residence Country*", options=list(country_options.keys()),
        index=list(country_options.keys()).index(
            st.session_state.get("employee_residence_select", "United States (US)")),
        key="employee_residence_select"
    )
    employee_residence = country_options[employee_residence_label]

    job_title = st.selectbox(
        "Job Title*", options=[
            "Data Engineer", "Data Scientist", "Machine Learning Engineer",
            "Data Analyst", "Data Architect"
        ],
        index=["Data Engineer", "Data Scientist", "Machine Learning Engineer", "Data Analyst", "Data Architect"]
        .index(st.session_state.get("job_title_select", "Data Engineer")),
        key="job_title_select"
    )

    # --- NÚT SUBMIT / RESET ---
    submitted = st.form_submit_button("PREDICT SALARY", type="primary")
    reset = st.form_submit_button("RESET")

# --- RESET FORM ---
if reset:
    st.session_state["reset_flag"] = True
    st.rerun()

# --- DỰ ĐOÁN ---
if submitted and model is not None and encoder is not None and scaler is not None:
    input_data = pd.DataFrame({
        'work_year': [work_year],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'employee_residence': [employee_residence],
        'remote_ratio': [int(remote_ratio)],
        'company_location': [company_location],
        'company_size': [company_size]
    })

    try:
        X, _, _ = encode_features(input_data, encoder=encoder, scaler=scaler, fit_encoder=False, fit_scaler=False)
        prediction = model.predict(X)[0]
        predicted_salary = round(prediction)

        st.success("Prediction Successful!")
        st.metric("**Predicted Salary (USD/year)**", f"${predicted_salary:,}", delta="Estimated result")
        with st.expander("📊 View Input Details"):
            st.dataframe(input_data, hide_index=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
