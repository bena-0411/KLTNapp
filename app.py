import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os

# 1. Cấu hình trang
st.set_page_config(page_title="DiaRisk 30 Pro", page_icon="🩺", layout="wide")

# 2. Tải mô hình và bộ tiền xử lý từ file joblib (dạng Dictionary)
@st.cache_resource
def load_model_bundle():
    data = joblib.load('kltn_diabetes_model_final.joblib')
    return data['model'], data['preprocessor'], data.get('optimal_threshold', 0.25)

try:
    model, preprocessor, optimal_threshold = load_model_bundle()
except Exception as e:
    st.error(f"Không thể tải file mô hình 'kltn_diabetes_model_final.joblib'. Lỗi: {e}")
    st.stop()

# File lưu lịch sử bệnh nhân
HISTORY_FILE = 'patient_history.csv'

# Sidebar: Lịch sử bệnh nhân
st.sidebar.header("📜 Lịch sử Bệnh nhân Đã đánh giá")
if os.path.exists(HISTORY_FILE):
    df_hist = pd.read_csv(HISTORY_FILE)
    search_term = st.sidebar.text_input("🔍 Tìm kiếm (Tên / Mã hồ sơ)")
    if search_term:
        df_hist = df_hist[
            df_hist['Họ tên'].astype(str).str.contains(search_term, case=False, na=False) |
            df_hist['Mã hồ sơ'].astype(str).str.contains(search_term, case=False, na=False)
        ]
    st.sidebar.dataframe(df_hist, use_container_width=True)
    
    if st.sidebar.button("🗑️ Xóa toàn bộ lịch sử"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.sidebar.success("Đã xóa lịch sử thành công!")
            st.rerun()
else:
    st.sidebar.info("Chưa có lịch sử bệnh nhân nào được lưu.")

# 3. Giao diện tiêu đề
st.title("🩺 DiaRisk 30 Pro - Hệ thống Dự báo Tái nhập viện")
st.markdown("Hệ thống Hỗ trợ Ra quyết định Lâm sàng (CDSS) dự báo nguy cơ tái nhập viện trong 30 ngày ở bệnh nhân đái tháo đường.")
st.markdown("---")

# 4. Form nhập liệu mô phỏng chuẩn giao diện Thesis (Hình 3.6)
with st.form("patient_form"):
    st.subheader("📝 Thông tin Định danh & Lâm sàng Bệnh nhân")

    # Thông tin định danh (như mô tả trong Mục 3.8 và Hình 3.6)
    id_col1, id_col2 = st.columns(2)
    with id_col1:
        patient_name = st.text_input("Họ và tên bệnh nhân (Bắt buộc)", placeholder="Nhập họ và tên...")
    with id_col2:
        patient_code = st.text_input("Mã Hồ sơ Bệnh án (Bắt buộc)", placeholder="Nhập mã hồ sơ (ví dụ: BN-999)...")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Hành chính & Nhập viện**")
        age = st.number_input("Nhóm tuổi / Tuổi (năm)", min_value=0, max_value=120, value=60)
        gender = st.selectbox("Giới tính", ["Female", "Male", "Other"])
        discharge_destination = st.selectbox("Tình trạng xuất viện (Discharge ID)", ["Home", "Nursing_Facility", "Other"])

    with col2:
        st.markdown("**2. Tiền sử & Sử dụng Dịch vụ Y tế**")
        length_of_stay = st.number_input("Số ngày nằm viện đợt hiện tại", min_value=1, max_value=30, value=5)
        medication_count = st.number_input("Tổng số loại thuốc đang sử dụng", min_value=1, max_value=50, value=10)
        diabetes = st.selectbox("Có chỉ định dùng thuốc ĐTĐ?", ["Yes", "No"])
        hypertension = st.selectbox("Có mắc Cao huyết áp?", ["Yes", "No"])

    with col3:
        st.markdown("**3. Chỉ số Sinh hiệu & Cận lâm sàng**")
        bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        cholesterol = st.number_input("Chỉ số Mỡ máu (Cholesterol)", min_value=50.0, max_value=400.0, value=200.0)
        systolic = st.number_input("Huyết áp Tâm thu (Systolic)", min_value=70.0, max_value=200.0, value=120.0)
        diastolic = st.number_input("Huyết áp Tâm trương (Diastolic)", min_value=40.0, max_value=130.0, value=80.0)

    submit_btn = st.form_submit_button("🔍 Thực hiện Đánh giá & Lưu hồ sơ")

# 5. Xử lý dự báo
if submit_btn:
    if not patient_name or not patient_code:
        st.warning("⚠️ Vui lòng nhập đầy đủ **Họ và tên bệnh nhân** và **Mã Hồ sơ Bệnh án** trước khi đánh giá!")
    else:
        # Chuẩn bị DataFrame đúng 11 đặc trưng đầu vào khớp với Pipeline
        input_data = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'cholesterol': cholesterol,
            'bmi': bmi,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'medication_count': medication_count,
            'length_of_stay': length_of_stay,
            'discharge_destination': discharge_destination,
            'systolic': systolic,
            'diastolic': diastolic
        }])

        try:
            # Biến đổi qua bộ preprocessor của ColumnTransformer
            X_processed = preprocessor.transform(input_data)
            
            # Dự báo xác suất lớp 1 (Tái nhập viện)
            prob = model.predict_proba(X_processed)[0][1]
            risk_percentage = prob * 100

            # Lưu vào lịch sử (CSV)
            risk_level = 'CAO' if risk_percentage >= 50 else ('TRUNG BÌNH' if risk_percentage >= 25 else 'THẤP')
            new_record = pd.DataFrame([{
                'Thời gian': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Họ tên': patient_name,
                'Mã hồ sơ': patient_code,
                'Tuổi': age,
                'Giới tính': gender,
                'Xác suất (%)': round(risk_percentage, 2),
                'Nguy cơ': risk_level
            }])
            
            if os.path.exists(HISTORY_FILE):
                df_old = pd.read_csv(HISTORY_FILE)
                df_new = pd.concat([df_old, new_record], ignore_index=True)
            else:
                df_new = new_record
            df_new.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')

            st.markdown("---")
            # Hiển thị tiêu đề báo cáo giống Hình 3.7 trong Khóa luận
            st.markdown(f"### 📊 Báo cáo Lâm sàng: **{patient_name}** (Mã hồ sơ: `{patient_code}`)")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                # Vẽ Gauge Chart bằng Plotly (Thước đo rủi ro)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_percentage,
                    title = {'text': "Xác suất Tái nhập viện (< 30 ngày)", 'font': {'size': 20}},
                    number = {'suffix': "%", 'font': {'size': 45}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "gold"},
                            {'range': [50, 100], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': risk_percentage
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Cảnh báo phân tầng rủi ro y khoa
                st.markdown("**Khuyến nghị chuyên môn:**")
                if risk_percentage >= 50:
                    st.error(f"🚨 **Nguy cơ CAO:** Bệnh nhân {patient_name} có xác suất tái nhập viện cấp tính cao ({risk_percentage:.1f}%). Cần theo dõi sát sao, tư vấn kỹ đơn thuốc xuất viện và hẹn tái khám sớm trong vòng 3-7 ngày.")
                elif risk_percentage >= 25:
                    st.warning(f"⚠️ **Nguy cơ TRUNG BÌNH:** Bệnh nhân có mức rủi ro đáng lưu ý ({risk_percentage:.1f}%). Cần hướng dẫn kỹ chế độ dinh dưỡng, tuân thủ điều trị ngoại trú và dặn dò các dấu hiệu cảnh báo.")
                else:
                    st.success(f"✅ **Nguy cơ THẤP:** Bệnh nhân ở mức an toàn ({risk_percentage:.1f}%). Đủ điều kiện xuất viện theo phác đồ tiêu chuẩn.")
                    
                # Các chỉ số cần lưu ý tự động
                st.markdown("##### 📌 Các chỉ số cần lưu ý:")
                if length_of_stay >= 7:
                    st.write(f"- Thời gian nằm viện kéo dài ({length_of_stay} ngày).")
                if medication_count >= 12:
                    st.write(f"- Bệnh nhân sử dụng nhiều loại thuốc kết hợp (đa trị liệu: {medication_count} loại).")
                if age >= 65:
                    st.write(f"- Bệnh nhân cao tuổi ({age} tuổi).")
                if systolic >= 140 or diastolic >= 90:
                    st.write(f"- Huyết áp cao bất thường ({int(systolic)}/{int(diastolic)} mmHg).")

            with res_col2:
                st.markdown("#### 🔬 Giải thích Khuyến nghị (Explainable AI bằng SHAP)")
                st.caption("Biểu đồ thể hiện mức độ tác động của từng đặc trưng đẩy xác suất rủi ro Lên cao (Đỏ) hoặc Xuống thấp (Xanh).")
                
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_processed)
                    feature_names = preprocessor.get_feature_names_out()
                    
                    val = shap_values.values
                    if len(val.shape) == 3:
                        class_idx = 1 if val.shape[2] > 1 else 0
                        sv = val[0, :, class_idx]
                        bv = explainer.expected_value
                        if isinstance(bv, (list, tuple, np.ndarray)):
                            bv = bv[class_idx]
                    else:
                        sv = val[0]
                        bv = explainer.expected_value
                        if isinstance(bv, (list, tuple, np.ndarray)):
                            bv = bv[0]

                    exp = shap.Explanation(
                        values=sv,
                        base_values=bv,
                        data=X_processed[0],
                        feature_names=feature_names
                    )
                    
                    fig_shap, ax_shap = plt.subplots(figsize=(6, 5))
                    shap.plots.waterfall(exp, show=False)
                    st.pyplot(fig_shap)
                except Exception as ex:
                    st.warning(f"Không thể hiển thị biểu đồ SHAP chi tiết: {ex}")

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý dữ liệu và dự báo: {e}")

# Footer disclaimer theo yêu cầu khóa luận
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 13px; font-style: italic;'>"
    "⚠️ <strong>Lưu ý:</strong> Công cụ hỗ trợ nghiên cứu và tham khảo lâm sàng, không thay thế chẩn đoán chuyên môn của bác sĩ."
    "</p>",
    unsafe_allow_html=True
)