"""
Trang 2 - Data Analysis: phân tích dữ liệu và trực quan hóa.
"""

import pandas as pd
import streamlit as st

from src.prediction import load_data
from src.ui import page_header
from src.visualization import (
    vi_name, vi_label, TIME_COLS,
    plot_histogram, plot_boxplot, plot_corr_bar, plot_corr_heatmap,
)
from preprocessing import load_raw_data, NEAR_CONSTANT_COLS

page_header("Phân tích dữ liệu và trực quan hóa", "Xem dữ liệu, so sánh hai nhóm ví và các bước tiền xử lý.")

df, df_train, df_test, features = load_data()
labels = df["FLAG"].map({0: "Bình thường", 1: "Gian lận"})

tab1, tab2, tab3, tab4 = st.tabs(["Tổng quan dữ liệu", "Phân phối thuộc tính", "Tương quan", "Tiền xử lý"])

# ---------- Tab 1: bảng dữ liệu ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số ví", f"{len(df):,}")
    c2.metric("Số thuộc tính số", len(features))
    c3.metric("Ví gian lận", f"{int(df['FLAG'].sum()):,}")
    st.caption("Thuộc tính số không tính địa chỉ ví và nhãn.")

    with st.container(border=True):
        st.markdown("**10 dòng đầu**")
        st.dataframe(df.head(10), width="stretch")

    with st.container(border=True):
        st.markdown("**Thống kê mô tả theo từng nhóm ví**")
        group = st.radio("Chọn nhóm", ["Bình thường", "Gian lận"], horizontal=True, label_visibility="collapsed")
        st.dataframe(df[labels == group][features].describe().T, width="stretch")

# ---------- Tab 2: phân phối ----------
with tab2:
    feat = st.selectbox("Chọn thuộc tính để xem:", features, format_func=vi_label)
    values = df[feat].astype(float)
    if feat in TIME_COLS:  # phút -> ngày
        values = values / 1440

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.plotly_chart(plot_histogram(values, labels), width="stretch")
    with c2, st.container(border=True):
        st.plotly_chart(plot_boxplot(values, labels), width="stretch")
    st.caption("Giá trị được biến đổi log (giữ dấu) để các giá trị rất lớn không làm biểu đồ bị dồn về một phía.")

    with st.container(border=True):
        st.markdown("**Trung bình / trung vị theo nhóm**")
        summary = pd.DataFrame({
            "Trung bình": values.groupby(labels).mean(),
            "Trung vị": values.groupby(labels).median(),
        })
        st.dataframe(summary, width="stretch")

# ---------- Tab 3: tương quan ----------
with tab3:
    corr = df[features + ["FLAG"]].corr()["FLAG"].drop("FLAG")
    k = st.slider("Số thuộc tính tương quan mạnh nhất cần hiển thị", 5, 20, 10)
    top = corr.abs().sort_values(ascending=False).head(k).index

    c1, c2 = st.columns([1, 1.2])
    with c1, st.container(border=True):
        st.plotly_chart(plot_corr_bar(corr[top]), width="stretch")
    with c2, st.container(border=True):
        st.markdown("**Ma trận tương quan giữa các thuộc tính hàng đầu**")
        st.plotly_chart(plot_corr_heatmap(df[list(top)].corr()), width="stretch")

# ---------- Tab 4: tóm tắt tiền xử lý ----------
with tab4:
    raw = load_raw_data()
    c1, c2, c3 = st.columns(3)
    c1.metric("Dữ liệu thô", f"{raw.shape[0]:,} dòng x {raw.shape[1]} cột")
    c2.metric("Sau khi làm sạch", f"{df.shape[0]:,} dòng x {df.shape[1]} cột")
    c3.metric("Số dòng bị loại (trùng lặp)", f"{raw.shape[0] - df.shape[0]:,}")
    st.info("Ứng dụng và mô hình đọc trực tiếp file **data/silver_ml_ready.csv** (bản đã làm sạch từ dữ liệu thô bằng các bước dưới đây).")

    left, right = st.columns([1.4, 1])
    with left, st.container(border=True):
        st.markdown(
            "**Các bước làm sạch:**\n"
            "1. Bỏ cột chỉ mục thừa (`Unnamed: 0`, `Index`).\n"
            "2. Bỏ các cột chỉ có 1 giá trị duy nhất (hằng số).\n"
            "3. Bỏ các cột chữ (tên token) vì mô hình chỉ dùng số; giữ lại cột `Address` để tra cứu.\n"
            "4. Điền 0 vào ô trống, xóa các dòng trùng lặp.\n"
            f"5. Bỏ {len(NEAR_CONSTANT_COLS)} cột gần như toàn số 0.\n"
            "6. Khi huấn luyện, áp dụng biến đổi `sign(x)·log(1+|x|)` để nén các giá trị cực lớn."
        )
    with right, st.container(border=True):
        st.markdown("**Chia dữ liệu:** 80% Train / 20% Test (giữ nguyên tỷ lệ gian lận ở hai tập).")
        split_table = pd.DataFrame({
            "Tập": ["Train", "Test"],
            "Số ví": [len(df_train), len(df_test)],
            "Ví gian lận": [int(df_train["FLAG"].sum()), int(df_test["FLAG"].sum())],
        })
        st.dataframe(split_table, hide_index=True, width="stretch")