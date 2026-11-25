"""
Trang 4 - Model Comparison: so sánh các mô hình Machine Learning trên tập Test.
"""

import streamlit as st

from src.prediction import load_metrics, load_test_predictions, MODEL_NAMES, THRESHOLD
from src.ui import page_header
from src.visualization import plot_metric_bar, plot_roc, plot_pr

page_header("So sánh các mô hình Machine Learning", f"Kết quả trên tập Test (ngưỡng phân loại = {THRESHOLD}).")

metrics = load_metrics()
preds = load_test_predictions()

selected = st.multiselect("Chọn mô hình cần so sánh:", MODEL_NAMES, default=MODEL_NAMES)
if not selected:
    st.warning("Hãy chọn ít nhất 1 mô hình.")
    st.stop()

sub = metrics[metrics["Model"].isin(selected)]

tab1, tab2, tab3 = st.tabs(["Chỉ số", "Đường cong ROC và PR", "Kết quả bắt gian lận"])

# ---------- 1. Bảng số liệu + biểu đồ cột ----------
with tab1:
    with st.container(border=True):
        st.dataframe(sub[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]],
                     width="stretch", hide_index=True)
    with st.container(border=True):
        st.plotly_chart(plot_metric_bar(sub, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]),
                        width="stretch")

# ---------- 2. Đường cong ROC và Precision-Recall ----------
with tab2:
    probs = {m: preds[f"prob_{m}"].values for m in selected}
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.plotly_chart(plot_roc(preds["y_true"].values, probs), width="stretch")
    with c2, st.container(border=True):
        st.plotly_chart(plot_pr(preds["y_true"].values, probs), width="stretch")

# ---------- 3. Ma trận nhầm lẫn bằng lời ----------
with tab3:
    cols = st.columns(len(sub))
    for col, (_, r) in zip(cols, sub.iterrows()):
        with col.container(border=True):
            st.markdown(f"**{r['Model']}**")
            st.metric("Bắt đúng ví gian lận", f"{r['TP']}/{r['TP'] + r['FN']}", f"bỏ sót {r['FN']}", delta_color="off", delta_arrow="off")
            st.metric("Báo nhầm ví sạch", f"{r['FP']}/{r['FP'] + r['TN']}")

st.info(
    "**Nhận xét:** Dữ liệu chỉ có khoảng 18% ví gian lận nên *Accuracy* dễ gây hiểu nhầm. "
    "Hãy nhìn *Recall* (bắt được bao nhiêu ví gian lận), *Precision* (cảnh báo đúng bao nhiêu) và *PR-AUC*."
)