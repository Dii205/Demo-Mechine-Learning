import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="So sánh Thuật toán ML", layout="wide")

st.title("🧪 Demo So Sánh 3 Thuật Toán Học Máy")
st.markdown("""
Ứng dụng này so sánh hiệu suất của **Decision Tree**, **Random Forest** và **SVM** trên bộ dữ liệu chẩn đoán ung thư vú (Breast Cancer).
""")
st.divider()

# --- 1. CHUẨN BỊ DỮ LIỆU ---
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Chuẩn hóa dữ liệu (quan trọng cho SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. THANH ĐIỀU KHIỂN (SIDEBAR) ---
st.sidebar.header("Tùy chỉnh tham số")
st.sidebar.subheader("Random Forest")
n_trees = st.sidebar.slider("Số lượng cây", 10, 200, 100)

st.sidebar.subheader("SVM")
svm_kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])

# --- 3. HUẤN LUYỆN MÔ HÌNH ---
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=n_trees, random_state=42),
    "SVM": SVC(kernel=svm_kernel, probability=True, random_state=42)
}

results = []

# Tạo 3 cột để hiển thị nhanh kết quả Accuracy
cols = st.columns(3)

for idx, (name, model) in enumerate(models.items()):
    # Huấn luyện
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Tính toán chỉ số
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results.append({
        "Thuật toán": name, 
        "Accuracy": round(acc, 4),
        "Precision": round(pre, 4),
        "Recall": round(rec, 4)
    })
    
    with cols[idx]:
        st.metric(label=f"Độ chính xác {name}", value=f"{acc:.2%}")

# --- 4. HIỂN THỊ KẾT QUẢ ---
df_results = pd.DataFrame(results)

st.subheader("📊 Bảng so sánh chi tiết")
st.table(df_results)

# Vẽ biểu đồ bằng Plotly
st.subheader("📈 Biểu đồ trực quan")
fig = px.bar(
    df_results, 
    x="Thuật toán", 
    y="Accuracy", 
    color="Thuật toán",
    text_auto='.2%',
    title="So sánh Accuracy giữa các mô hình"
)
st.plotly_chart(fig, use_container_width=True)

st.success("✅ Mô hình đã được huấn luyện thành công trên tập dữ liệu chuẩn!")