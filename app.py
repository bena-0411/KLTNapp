import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 1. Cấu hình trang
st.set_page_config(page_title="DiaRisk 30 Pro", page_icon="🩺", layout="wide")

# 2. Tải mô hình
@st.cache_resource
def load_model():
    # Tải file mô hình đã được gửi
    return joblib.load('kltn_diabetes_model_final.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Không tìm thấy file mô hình 'kltn_diabetes_model_final.joblib'. Vui lòng đảm bảo file nằm cùng thư mục với app.py. Lỗi: {e}")
    st.stop()

# 3. Giao diện tiêu đề
st.title("🩺 DiaRisk 30 Pro - Hệ thống Dự báo Tái nhập viện")
st.markdown("Hệ thống Hỗ trợ Ra quyết định Lâm sàng dự báo nguy cơ tái nhập viện trong 30 ngày cho bệnh nhân dựa trên mô hình Học máy (Extra Trees Classifier).")
st.markdown("---")

# 4. Form nhập liệu
with st.form("patient_form"):
    st.subheader("📝 Nhập Thông tin Lâm sàng Bệnh nhân")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Hành chính & Nhập viện**")
        age = st.number_input("Tuổi (năm)", min_value=0, max_value=120, value=60)
        gender = st.selectbox("Giới tính", ["Female", "Male", "Other"])
        discharge_destination = st.selectbox("Điểm đến sau xuất viện", ["Home", "Nursing_Facility", "Other"])

    with col2:
        st.markdown("**2. Thông tin Điều trị & Bệnh nền**")
        length_of_stay = st.number_input("Số ngày nằm viện đợt này", min_value=1, max_value=30, value=5)
        medication_count = st.number_input("Tổng số loại thuốc đang dùng", min_value=1, max_value=50, value=10)
        diabetes = st.selectbox("Bệnh nhân có mắc Đái tháo đường?", ["Yes", "No"])
        hypertension = st.selectbox("Bệnh nhân có mắc Cao huyết áp?", ["Yes", "No"])

    with col3:
        st.markdown("**3. Chỉ số Sinh hiệu & Xét nghiệm**")
        bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        cholesterol = st.number_input("Chỉ số Mỡ máu (Cholesterol)", min_value=50.0, max_value=400.0, value=200.0)
        systolic = st.number_input("Huyết áp Tâm thu (Systolic)", min_value=70.0, max_value=200.0, value=120.0)
        diastolic = st.number_input("Huyết áp Tâm trương (Diastolic)", min_value=40.0, max_value=130.0, value=80.0)

    submit_btn = st.form_submit_button("🔍 Thực hiện Đánh giá Rủi ro")

# 5. Xử lý dự báo
if submit_btn:
    # Tạo DataFrame từ input khớp với 11 biến của dữ liệu mới
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
        # Lấy xác suất dự báo lớp 1 (Tái nhập viện)
        prob = model.predict_proba(input_data)[0][1]
        risk_percentage = prob * 100

        st.markdown("---")
        st.header("📊 Báo cáo Phân tầng Rủi ro")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Vẽ Gauge Chart bằng Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_percentage,
                title = {'text': "Xác suất Tái nhập viện (< 30 ngày)", 'font': {'size': 24}},
                number = {'suffix': "%", 'font': {'size': 50}},
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

            # Cảnh báo y khoa tự động
            st.info("**Khuyến nghị chuyên môn:**")
            if risk_percentage >= 50:
                st.error("🚨 **Nguy cơ CAO:** Bệnh nhân có xác suất tái nhập viện cấp tính cao. Cần theo dõi sát sao, cân nhắc giữ lại lưu viện thêm hoặc sắp xếp lịch tái khám sớm trong vòng 3-7 ngày sau xuất viện.")
            elif risk_percentage >= 25:
                st.warning("⚠️ **Nguy cơ TRUNG BÌNH:** Cần tư vấn kỹ phác đồ dùng thuốc, hướng dẫn chế độ dinh dưỡng và các dấu hiệu nhận biết biến chứng khi điều trị ngoại trú ở nhà.")
            else:
                st.success("✅ **Nguy cơ THẤP:** Bệnh nhân đủ điều kiện an toàn để xuất viện theo phác đồ tiêu chuẩn.")
                
            # Trích xuất lý do tự động dựa trên input
            st.markdown("**Các chỉ số cần lưu ý:**")
            if length_of_stay >= 7:
                st.write(f"- Thời gian nằm viện kéo dài ({length_of_stay} ngày).")
            if medication_count >= 15:
                st.write(f"- Bệnh nhân đang sử dụng đa trị liệu (nhiều loại thuốc kết hợp: {medication_count} loại).")
            if age >= 65:
                st.write(f"- Bệnh nhân cao tuổi ({age} tuổi).")

        with res_col2:
            st.markdown("#### Khả năng giải thích mô hình (SHAP Waterfall)")
            st.caption("Biểu đồ thể hiện mức độ tác động của từng đặc trưng đẩy rủi ro Lên cao (Đỏ) hoặc Xuống thấp (Xanh).")
            
            # Xử lý SHAP cho Pipeline
            try:
                # Nếu model là 1 Pipeline chứa ColumnTransformer và Classifier
                if hasattr(model, 'named_steps'):
                    preprocessor = model.steps[0][1]
                    classifier = model.steps[-1][1]
                    
                    # Transform data
                    X_processed = preprocessor.transform(input_data)
                    
                    # Khởi tạo SHAP TreeExplainer
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer(X_processed)
                    
                    # Lấy tên cột sau khi One-Hot Encoding
                    feature_names = preprocessor.get_feature_names_out()
                    
                    # Tạo đối tượng Explanation để vẽ Waterfall
                    exp = shap.Explanation(values=shap_values[0].values, 
                                           base_values=shap_values[0].base_values, 
                                           data=X_processed[0], 
                                           feature_names=feature_names)
                    
                    fig_shap, ax_shap = plt.subplots(figsize=(6, 5))
                    shap.plots.waterfall(exp, show=False)
                    st.pyplot(fig_shap)
                else:
                    st.warning("Cấu trúc mô hình không phải là Pipeline chuẩn. Không thể render SHAP tự động.")
            except Exception as e:
                st.error(f"Lỗi hiển thị SHAP: {e}")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình dự báo: {e}")
