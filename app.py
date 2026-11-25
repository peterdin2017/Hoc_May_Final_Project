"""
app.py - Điểm vào của ứng dụng Streamlit: thanh điều hướng + trang chủ.
Các trang chức năng nằm trong thư mục pages/.

Chạy:  streamlit run app.py
"""

import streamlit as st

from src.prediction import load_data, MODEL_NAMES, THRESHOLD, LOW_LIMIT
from src.ui import inject_css, hero

st.set_page_config(
    page_title="Phát hiện ví Ethereum gian lận",
    page_icon=":material/shield:",
    layout="wide",
)
inject_css()


def home():
    hero(
        "Hệ thống phát hiện ví Ethereum gian lận",
        "Ứng dụng dùng Học máy phân tích lịch sử giao dịch của một địa chỉ ví Ethereum "
        "để ước lượng <b>xác suất ví đó có dấu hiệu gian lận</b> (phishing, lừa đảo, rửa tiền).",
        chips=MODEL_NAMES,
    )

    df, df_train, df_test, features = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số ví (sau khi làm sạch)", f"{len(df):,}")
    c2.metric("Số thuộc tính", len(features))
    c3.metric("Tỷ lệ ví gian lận", f"{df['FLAG'].mean() * 100:.1f}%")
    c4.metric("Số mô hình", len(MODEL_NAMES))

    st.subheader("Bạn muốn làm gì?")
    cards = [
        ("card_predict", "Kiểm tra rủi ro ví", "pages/3_Prediction.py", ":material/search:",
         "Kiểm tra rủi ro ví, đối chiếu đặc trưng và đề xuất giải pháp an toàn. <b>Bắt đầu từ đây.</b>"),
        ("card_dash", "Xem tổng quan", "pages/1_Dashboard.py", ":material/dashboard:",
         "Số liệu tổng quan của dữ liệu và mô hình."),
        ("card_data", "Phân tích dữ liệu", "pages/2_Data_Analysis.py", ":material/bar_chart:",
         "Phân tích, trực quan hóa và tiền xử lý dữ liệu."),
        ("card_model", "So sánh mô hình", "pages/4_Model_Comparison.py", ":material/compare_arrows:",
         "So sánh chi tiết các mô hình Machine Learning."),
    ]
    for col, (key, title, path, icon, text) in zip(st.columns(4), cards):
        with col.container(border=True, key=key):
            st.markdown(f'<div class="card-title">{title}</div><div class="card-text">{text}</div>',
                        unsafe_allow_html=True)
            st.page_link(path, label="Mở trang", icon=icon)

    st.subheader("Cách đọc mức rủi ro")
    st.markdown(
        '<div class="legend-row">'
        f'<span class="dot" style="background:#16A34A"></span><b>THẤP</b>: xác suất dưới {LOW_LIMIT * 100:.0f}%<br>'
        f'<span class="dot" style="background:#D97706"></span><b>CẦN XEM XÉT</b>: từ {LOW_LIMIT * 100:.0f}% đến dưới {THRESHOLD * 100:.0f}%<br>'
        f'<span class="dot" style="background:#DC2626"></span><b>CAO (nghi ngờ gian lận)</b>: từ {THRESHOLD * 100:.0f}% trở lên'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Lưu ý: kết quả chỉ là xác suất thống kê từ mô hình, không phải kết luận pháp lý.")


pg = st.navigation([
    st.Page(home, title="Trang chủ", icon=":material/home:", default=True),
    st.Page("pages/1_Dashboard.py", title="Tổng quan", icon=":material/dashboard:"),
    st.Page("pages/2_Data_Analysis.py", title="Phân tích dữ liệu", icon=":material/bar_chart:"),
    st.Page("pages/3_Prediction.py", title="Dự đoán", icon=":material/search:"),
    st.Page("pages/4_Model_Comparison.py", title="So sánh mô hình", icon=":material/compare_arrows:"),
])
pg.run()