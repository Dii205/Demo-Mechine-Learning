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
st.set_page_config(page_title="ML Comparison Tool", layout="centered")

st.title("🚀 Hệ Thống Demo & So Sánh Thuật Toán ML")
st.markdown("---")

# 2. Thanh điều khiển (Sidebar)
st.sidebar.header("⚙️ Cấu hình mô hình")
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu (CSV)", type=["csv"])

# Tham số Random Forest
st.sidebar.subheader("🌲 Random Forest")
n_trees = st.sidebar.slider("Số lượng cây", 1, 200, 100)

# Tham số SVM
st.sidebar.subheader("🧠 SVM (Support Vector Machine)")
svm_kernel = st.sidebar.selectbox("Chọn Kernel", ["rbf", "linear", "poly", "sigmoid"])
svm_c = st.sidebar.slider("Chỉ số C (Regularization)", 0.01, 10.0, 1.0)

if uploaded_file is None:
    st.warning("⚠️ Vui lòng tải lên file CSV ở thanh bên trái để bắt đầu.")
    st.stop()

# Đọc và chia dữ liệu
df = pd.read_csv(uploaded_file)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa (Dành riêng cho SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Huấn luyện mô hình
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=n_trees, random_state=42),
    "SVM": SVC(kernel=svm_kernel, C=svm_c, probability=True, random_state=42)
}

trained_models = {}
results = []

for name, model in models.items():
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
    else:
        # DT và RF dùng dữ liệu thô
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
    
    trained_models[name] = model
    results.append({"Thuật toán": name, "Accuracy": acc})

# 4. Hiển thị Kết quả huấn luyện
st.subheader("📊 1. Kết quả so sánh độ chính xác")
df_res = pd.DataFrame(results)
st.table(df_res)

fig = px.bar(df_res, x="Thuật toán", y="Accuracy", color="Thuật toán", 
             text_auto='.2%', title=f"Hiệu suất mô hình (SVM Kernel: {svm_kernel})")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 5. Dự đoán dữ liệu mới
st.subheader("🔮 2. Dự đoán dựa trên thông số nhập vào")
input_data = []
cols = st.columns(3)
for i, feature in enumerate(X.columns):
    with cols[i % 3]:
        val = st.number_input(f"{feature}", value=float(X[feature].mean()))
        input_data.append(val)

if st.button("🔍 Đưa ra kết quả dự đoán"):
    st.write("### Kết quả từ 3 thuật toán:")
    res_cols = st.columns(3)
    
    for idx, (name, model) in enumerate(trained_models.items()):
        if name == "SVM":
            new_data_input = scaler.transform([input_data])
            prediction = model.predict(new_data_input)[0]
        else:
            prediction = model.predict([input_data])[0]
        
        status = "Bệnh" if prediction == 1 else "Không bệnh"
        
        with res_cols[idx]:
            st.info(f"**{name}**")
            if prediction == 1:
                st.error(f"Kết quả: **{status}**")
            else:
                st.success(f"Kết quả: **{status}**")
