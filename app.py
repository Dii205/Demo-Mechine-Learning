import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Cấu hình giao diện
st.set_page_config(page_title="ML Prediction Tool", layout="centered")

st.title("🚀 Demo So Sánh & Dự Đoán Thuật Toán ML")
st.markdown("---")

# 2. Upload file (Bắt buộc)
st.sidebar.header("📥 Quản lý dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu (CSV) để bắt đầu", type=["csv"])

if uploaded_file is None:
    st.warning("⚠️ Vui lòng tải lên file CSV ở thanh bên trái để sử dụng ứng dụng.")
    st.stop() # Dừng app nếu chưa có file

# Đọc dữ liệu
df = pd.read_csv(uploaded_file)
st.success("Đã tải file thành công!")

# Tiền xử lý dữ liệu
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Chia dữ liệu và chuẩn hóa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Huấn luyện mô hình
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

trained_models = {}
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    results.append({"Thuật toán": name, "Accuracy": acc})

# 4. Hiển thị bảng và biểu đồ (Xếp dọc theo yêu cầu)
st.subheader("📊 1. Kết quả so sánh độ chính xác")
df_res = pd.DataFrame(results)

# Hiển thị bảng trước
st.table(df_res)

# Biểu đồ nằm xuống dòng dưới
fig = px.bar(df_res, x="Thuật toán", y="Accuracy", color="Thuật toán", 
             text_auto='.2%', title="Biểu đồ so sánh Accuracy")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 5. Nhập dữ liệu mới để dự đoán
st.subheader("🔮 2. Nhập thông số mới để dự đoán")

input_data = []
cols = st.columns(3) # Chia ô nhập liệu làm 3 cột cho đẹp
for i, feature in enumerate(X.columns):
    with cols[i % 3]:
        val = st.number_input(f"{feature}", value=float(X[feature].mean()))
        input_data.append(val)

if st.button("🔍 Đưa ra kết quả dự đoán"):
    new_data_scaled = scaler.transform([input_data])
    
    st.write("### Kết quả từ 3 thuật toán:")
    res_cols = st.columns(3)
    
    for idx, (name, model) in enumerate(trained_models.items()):
        prediction = model.predict(new_data_scaled)[0]
        
        # Chuyển đổi số 0/1 thành chữ
        if prediction == 1:
            status = "Bệnh"
            color = "error" # Màu đỏ/cam
        else:
            status = "Không bệnh"
            color = "success" # Màu xanh lá
            
        with res_cols[idx]:
            st.info(f"**{name}**")
            if prediction == 1:
                st.error(f"Kết quả: **{status}**")
            else:
                st.success(f"Kết quả: **{status}**")
