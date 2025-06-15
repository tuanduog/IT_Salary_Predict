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
st.set_page_config(page_title="Dự báo lương IT", layout="centered")
st.title("PHÂN TÍCH DỰ BÁO MỨC LƯƠNG NGÀNH IT")
st.markdown("""*Sử dụng mô hình Hồi quy Tuyến tính*""")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('notebook/model.pkl')
        encoder = joblib.load('notebook/encoder.pkl')
        return model, encoder
    except Exception as e:
        st.error(f"Lỗi khi tải model: {str(e)}")
        return None, None

model, encoder = load_model()

# --- USE CASE 1: NHẬP DỮ LIỆU ---
with st.form(key="salary_prediction_form"):
    st.subheader("1. Nhập thông tin dự báo")

    col1, col2 = st.columns(2)

    with col1:
        work_year = st.number_input(
            "Năm làm việc*",
            min_value=2020,
            max_value=2025,
            value=2024,
            key="work_year_input"
        )
        remote_ratio = st.selectbox(
            "Tỷ lệ làm việc từ xa*",
            options=["0", "50", "100"],
            index=1,
            key="remote_ratio_select"
        )

    with col2:
        experience_level = st.selectbox(
            "Trình độ kinh nghiệm*",
            options=["EN", "MI", "SE", "EX"],
            index=2,
            key="experience_level_select"
        )
        employment_type = st.selectbox(
            "Loại hình làm việc*",
            options=["FT", "PT", "CT", "FL"],
            key="employment_type_select"
        )

    company_size = st.selectbox(
        "Kích thước công ty*",
        options=["S", "M", "L"],
        index=1,
        key="company_size_select"
    )

    company_location = st.selectbox(
        "Vị trí công ty*",
        options=["US", "GB", "IN", "CA", "DE", "FR", "VN", "JP", "AU"],
        index=0,
        key="company_location_select"
    )

    employee_residence = st.selectbox(
        "Quốc gia cư trú*",
        options=["US", "GB", "IN", "CA", "DE", "FR", "VN", "JP", "AU"],
        index=0,
        key="employee_residence_select"
    )

    job_title = st.selectbox(
        "Chức danh công việc*",
        options=["Data Engineer", "Data Scientist", "Machine Learning Engineer", "Data Analyst", "Data Architect"],
        index=0,
        key="job_title_select"
    )

    # --- NÚT SUBMIT ---
    submitted = st.form_submit_button("DỰ BÁO LƯƠNG", type="primary")
    reset = st.form_submit_button("LÀM MỚI")

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

        st.success("DỰ BÁO THÀNH CÔNG!")
        st.metric(
            label="**Mức lương dự đoán (USD/năm)**",
            value=f"${predicted_salary:,}",
            delta="Kết quả ước lượng"
        )

        with st.expander("📊 Xem chi tiết đầu vào"):
            st.dataframe(input_data, hide_index=True)

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {str(e)}")