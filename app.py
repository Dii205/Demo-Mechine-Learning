import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# 1. Cấu hình giao diện
st.set_page_config(page_title="ML Prediction Tool", layout="wide")

st.title("🚀 Demo So Sánh & Dự Đoán với 3 Thuật Toán")
st.markdown("---")

# 2. Xử lý dữ liệu
st.sidebar.header("📥 Quản lý dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    raw_data = load_breast_cancer()
    df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    df['target'] = raw_data.target
    st.sidebar.info("Đang dùng dữ liệu mẫu (Ung thư vú)")

# Tiền xử lý
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

label_encoder = None
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Huấn luyện mô hình
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

trained_models = {}
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    y_pred = model.predict(X_test_scaled)
    results.append({"Thuật toán": name, "Accuracy": accuracy_score(y_test, y_pred)})

# 4. Hiển thị kết quả huấn luyện
st.subheader("📊 1. Kết quả so sánh độ chính xác")
df_res = pd.DataFrame(results)
col_table, col_chart = st.columns([1, 2])
with col_table:
    st.dataframe(df_res)
with col_chart:
    fig = px.bar(df_res, x="Thuật toán", y="Accuracy", color="Thuật toán", text_auto='.2%', height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 5. PHẦN NHẬP DỮ LIỆU MỚI ĐỂ DỰ ĐOÁN
st.subheader("🔮 2. Nhập thông số mới để dự đoán")
st.write("Nhập các giá trị dưới đây để xem 3 mô hình dự đoán kết quả thế nào:")

# Tạo các ô nhập liệu dựa trên số cột của dữ liệu
input_data = []
cols = st.columns(4) # Chia làm 4 cột cho gọn
for i, feature in enumerate(X.columns):
    with cols[i % 4]:
        # Lấy giá trị trung bình làm mặc định để người dùng dễ nhập
        val = st.number_input(f"{feature}", value=float(X[feature].mean()))
        input_data.append(val)

if st.button("🔍 Đưa ra kết quả dự đoán"):
    # Chuẩn hóa dữ liệu mới nhập theo scaler cũ
    new_data_scaled = scaler.transform([input_data])
    
    st.write("### Kết quả dự đoán từ 3 thuật toán:")
    res_cols = st.columns(3)
    
    for idx, (name, model) in enumerate(trained_models.items()):
        prediction = model.predict(new_data_scaled)
        
        # Giải mã nhãn nếu là chữ
        if label_encoder:
            final_label = label_encoder.inverse_transform(prediction)[0]
        else:
            final_label = prediction[0]
            
        with res_cols[idx]:
            st.info(f"**{name}**")
            st.success(f"Kết quả: **{final_label}**")
