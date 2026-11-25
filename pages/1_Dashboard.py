"""
Trang 1 - Dashboard (Tổng quan): các con số quan trọng nhất của dữ liệu và mô hình.
"""

import streamlit as st

from src.prediction import load_data, load_metrics, load_importance
from src.ui import page_header
from src.visualization import plot_label_pie, plot_metric_bar, plot_importance

page_header("Dashboard - Tổng quan hệ thống", "Các con số quan trọng nhất của dữ liệu và mô hình.")

df, df_train, df_test, features = load_data()
metrics = load_metrics()
importance = load_importance()

# ---------- 1. Các ô số liệu chính ----------
n_fraud = int(df["FLAG"].sum())
best = metrics.sort_values("PR_AUC", ascending=False).iloc[0]  # mô hình tốt nhất

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng số ví", f"{len(df):,}", "sau khi làm sạch", delta_color="off", delta_arrow="off")
c2.metric("Ví gian lận", f"{n_fraud:,}", f"{n_fraud / len(df) * 100:.1f}% tổng số ví", delta_color="off", delta_arrow="off")
c3.metric("Số ví trong tập Test", f"{len(df_test):,}", f"{len(df_test) / len(df) * 100:.0f}% tổng số ví", delta_color="off", delta_arrow="off")
c4.metric("Mô hình tốt nhất", best["Model"], f"PR-AUC {best['PR_AUC']:.3f}", delta_color="off", delta_arrow="off")

st.write("")

# ---------- 2. Biểu đồ ----------
left, right = st.columns([1, 1.4])
with left:
    with st.container(border=True):
        st.plotly_chart(plot_label_pie(len(df) - n_fraud, n_fraud), width="stretch")
with right:
    with st.container(border=True):
        st.plotly_chart(plot_metric_bar(metrics, ["Precision", "Recall", "F1", "PR_AUC"]), width="stretch")
        st.caption("So sánh các mô hình trên tập Test")

# ---------- 3. Thuộc tính quan trọng + bảng kết quả ----------
left, right = st.columns([1.1, 1])
with left:
    with st.container(border=True):
        st.markdown("**10 thuộc tính ảnh hưởng nhiều nhất đến dự đoán**")
        st.plotly_chart(plot_importance(importance, "Trung bình", top=10), width="stretch")
with right:
    with st.container(border=True):
        st.markdown("**Kết quả chi tiết (tập Test)**")
        st.dataframe(metrics[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]],
                     width="stretch", hide_index=True)
    st.info(
        "**Cách đọc:** *Recall* là tỷ lệ ví gian lận được phát hiện; *Precision* là tỷ lệ cảnh báo đúng. "
        "*PR-AUC* càng gần 1 thì mô hình càng tốt trên dữ liệu mất cân bằng."
    )