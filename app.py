import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ thống So sánh ML", layout="wide")

st.title("🚀 Hệ thống Demo & So sánh 3 Thuật toán ML")
st.markdown("Tải lên dữ liệu của bạn hoặc sử dụng dữ liệu mẫu để so sánh Decision Tree, Random Forest và SVM.")

# --- 1. XỬ LÝ DỮ LIỆU ---
st.sidebar.header("📥 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV của bạn", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Đã tải dữ liệu của bạn thành công!")
else:
    # Sử dụng dữ liệu mẫu nếu chưa có file
    raw_data = load_breast_cancer()
    df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    df['target'] = raw_data.target
    st.info("Đang sử dụng dữ liệu mẫu: Chẩn đoán Ung thư vú")

st.write("### Xem trước dữ liệu:", df.head(5))

# Tiền xử lý dữ liệu
X = df.iloc[:, :-1] # Lấy tất cả trừ cột cuối
y = df.iloc[:, -1]  # Lấy cột cuối làm nhãn

# Chuyển đổi nhãn chữ thành số nếu cần
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. CÀI ĐẶT THAM SỐ ---
st.sidebar.header("⚙️ Tùy chỉnh tham số")
n_trees = st.sidebar.slider("RF: Số lượng cây", 10, 200, 100)
svm_kernel = st.sidebar.selectbox("SVM: Kernel", ["linear", "rbf", "poly"])

# --- 3. HUẤN LUYỆN VÀ SO SÁNH ---
if st.button("🚀 Bắt đầu huấn luyện và So sánh"):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=n_trees, random_state=42),
        "SVM": SVC(kernel=svm_kernel, probability=True, random_state=42)
    }

    results = []
    
    with st.spinner('Đang tính toán...'):
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            results.append({
                "Thuật toán": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1-Score": f1_score(y_test, y_pred, average='weighted')
            })

    # Hiển thị bảng kết quả
    df_results = pd.DataFrame(results)
    st.write("### 📊 Kết quả đánh giá chi tiết")
    st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'))

    # Vẽ biểu đồ so sánh Accuracy
    st.write("### 📈 Biểu đồ so sánh độ chính xác (Accuracy)")
    fig = px.bar(df_results, x="Thuật toán", y="Accuracy", color="Thuật toán", 
                 text_auto='.2%', range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Giải thích cho bài tiểu luận
    st.write("---")
    st.write("### 💡 Nhận xét nhanh cho bài tiểu luận:")
    best_model = df_results.loc[df_results['Accuracy'].idxmax(), 'Thuật toán']
    st.markdown(f"- Thuật toán hoạt động tốt nhất trên dữ liệu này là: **{best_model}**")
    st.markdown("- **SVM** thường mạnh khi dữ liệu được chuẩn hóa tốt.")
    st.markdown("- **Random Forest** thường ổn định nhất nhờ sự kết hợp của nhiều cây quyết định.")
