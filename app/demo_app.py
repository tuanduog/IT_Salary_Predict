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
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, encoder = load_model()

# --- USE CASE 1: NHẬP DỮ LIỆU ---
with st.form(key="salary_prediction_form"):
    st.subheader("Enter Prediction Information")

    col1, col2 = st.columns(2)

    with col1:
        work_year = st.number_input(
            "Working Year*",
            min_value=2020,
            max_value=2025,
            value=2024,
            key="work_year_input"
        )
        remote_ratio = st.selectbox(
            "Remote Work Ratio*",
            options=["0", "50", "100"],
            index=1,
            key="remote_ratio_select"
        )

    with col2:
        # Label-value mapping
        experience_options = {
            "Fresher (EN)": "EN",
            "Junior (MI)": "MI",
            "Senior (SE)": "SE",
            "Executive (EX)": "EX"
        }

        employment_options = {
            "Full-time (FT)": "FT",
            "Part-time (PT)": "PT",
            "Contract (CT)": "CT",
            "Freelance (FL)": "FL"
        }

        experience_label = st.selectbox(
            "Experience Level*",
            options=list(experience_options.keys()),
            index=2,
            key="experience_level_select"
        )
        experience_level = experience_options[experience_label]

        employment_label = st.selectbox(
            "Employment Type*",
            options=list(employment_options.keys()),
            key="employment_type_select"
        )
        employment_type = employment_options[employment_label]

    company_size_options = {
        "Small (S)": "S",
        "Medium (M)": "M",
        "Large (L)": "L"
    }
    company_size_label = st.selectbox(
        "Company Size*",
        options=list(company_size_options.keys()),
        index=1,
        key="company_size_select"
    )
    company_size = company_size_options[company_size_label]

    country_options = {
        "United States (US)": "US",
        "United Kingdom (GB)": "GB",
        "India (IN)": "IN",
        "Canada (CA)": "CA",
        "Germany (DE)": "DE",
        "France (FR)": "FR",
        "Vietnam (VN)": "VN",
        "Japan (JP)": "JP",
        "Australia (AU)": "AU"
    }

    company_location_label = st.selectbox(
        "Company Location*",
        options=list(country_options.keys()),
        index=0,
        key="company_location_select"
    )
    company_location = country_options[company_location_label]

    employee_residence_label = st.selectbox(
        "Employee Residence Country*",
        options=list(country_options.keys()),
        index=0,
        key="employee_residence_select"
    )
    employee_residence = country_options[employee_residence_label]

    job_title = st.selectbox(
        "Job Title*",
        options=[
            "Data Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "Data Analyst",
            "Data Architect"
        ],
        index=0,
        key="job_title_select"
    )

    # --- NÚT SUBMIT / RESET ---
    submitted = st.form_submit_button("PREDICT SALARY", type="primary")
    reset = st.form_submit_button("RESET")

# --- XỬ LÝ USE CASE 2: LÀM MỚI ---
if reset:
    st.rerun()

# --- XỬ LÝ USE CASE 3: DỰ BÁO ---
if submitted and model is not None:
    input_data = pd.DataFrame({
        'work_year': [work_year],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'employee_residence': [employee_residence],
        'remote_ratio': [remote_ratio],
        'company_location': [company_location],
        'company_size': [company_size]
    })

    try:
        X, _ = encode_features(input_data, encoder, fit_encoder=False)
        prediction = model.predict(X)[0]
        predicted_salary = round(prediction)

        st.success("Prediction Successful!")
        st.metric(
            label="**Predicted Salary (USD/year)**",
            value=f"${predicted_salary:,}",
            delta="Estimated result"
        )

        with st.expander("📊 View Input Details"):
            st.dataframe(input_data, hide_index=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
