"""
Trang 3 - Prediction: dự đoán ví có gian lận hay không và đề xuất giải pháp an toàn.
3 cách nhập: (1) chọn ví mẫu trong tập Test, (2) nhập tay 6 thông số, (3) tải file CSV.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.prediction import (
    load_data, load_importance, predict_wallet, predict_many, normal_medians,
    build_manual_wallet, classify_risk, check_sufficient_data,
    THRESHOLD, ID_COL, TX_ETH_COL, TX_ERC20_COL,
)
from src.recommendation import (
    render_recommendations_box,
    summarize_case_for_csv, compare_with_groups, GENERAL_TIPS,
)
from src.ui import page_header, risk_badge
from src.visualization import TIME_COLS, plot_gauge

page_header(
    "Dự đoán rủi ro ví Ethereum & Đề xuất an toàn",
    "Chọn một cách nhập dữ liệu bên dưới, kết quả sẽ hiện ngay phía dưới.",
)

df, df_train, df_test, features = load_data()
top_features = load_importance().head(10)["feature"].tolist()


def show_result(probs, row):
    """Hiển thị xác suất dự đoán, đối chiếu mô hình, khuyến nghị đóng khung theo từng ví và giải thích đặc trưng."""
    prob = probs["XGBoost"]
    level, box = classify_risk(prob)

    # ---------- Kết quả chính: đồng hồ + mức rủi ro + đối chiếu mô hình ----------
    with st.container(border=True):
        left, right = st.columns([1, 1.15], vertical_alignment="center")
        with left:
            st.plotly_chart(plot_gauge(prob, THRESHOLD), width="stretch")
            st.caption(f"Vạch xám là ngưỡng cảnh báo ({THRESHOLD * 100:.0f}%).")
        with right:
            risk_badge(level, box)
            st.write("")
            st.markdown("**Đối chiếu với các mô hình khác:**")
            votes = 0
            for name, p in probs.items():
                votes += int(p >= THRESHOLD)
                st.progress(
                    float(min(max(p, 0.0), 1.0)),
                    text=f"{name}: {p * 100:.1f}% - " + ("Nghi ngờ gian lận" if p >= THRESHOLD else "Bình thường"),
                )
            st.caption(f"{votes}/{len(probs)} mô hình cảnh báo gian lận.")

    # ---------- Chi tiết ----------
    tab_act, tab_cmp = st.tabs(["Hành động được khuyến nghị", "Ví này giống nhóm nào hơn?"])
    with tab_act:
        render_recommendations_box(row, level, prob)
    with tab_cmp:
        st.caption(
            "So sánh giá trị của ví với trung vị của ví bình thường và ví gian lận trong tập Train "
            "(10 thuộc tính quan trọng nhất)."
        )
        table = compare_with_groups(row, df_train, top_features)
        n_fraud_like = int((table["Gần nhóm"] == "Gian lận").sum())
        st.write(f"Có **{n_fraud_like}/10** thuộc tính quan trọng của ví này gần với nhóm ví gian lận hơn.")
        st.dataframe(table, width="stretch", hide_index=True)


tab_sample, tab_manual, tab_csv = st.tabs(["Ví mẫu từ tập Test", "Nhập tay 6 thông số", "Tải file CSV"])

# ====================== CÁCH 1: VÍ MẪU TỪ TẬP TEST ======================
with tab_sample:
    st.info("Các ví này có nhãn thật (FLAG) và **không** dùng để huấn luyện mô hình, nên có thể dùng để kiểm tra.")

    if "chosen" not in st.session_state:
        st.session_state["chosen"] = int(df_test.index[0])

    b1, b2 = st.columns(2)
    if b1.button("Chọn ngẫu nhiên 1 ví gian lận", width="stretch", icon=":material/shuffle:"):
        st.session_state["chosen"] = int(np.random.choice(df_test[df_test["FLAG"] == 1].index))
    if b2.button("Chọn ngẫu nhiên 1 ví bình thường", width="stretch", icon=":material/shuffle:"):
        st.session_state["chosen"] = int(np.random.choice(df_test[df_test["FLAG"] == 0].index))

    options = df_test.index.tolist()
    chosen = st.selectbox(
        "Địa chỉ ví:",
        options,
        index=options.index(st.session_state["chosen"]),
        format_func=lambda i: df_test.loc[i, ID_COL],
    )
    st.session_state["chosen"] = chosen

    row = df_test.loc[chosen]
    k0, k1, k2, k3, k4 = st.columns(5)
    k0.metric("Nhãn thực tế", "Gian lận" if row["FLAG"] == 1 else "Bình thường")
    k1.metric("Tuổi hoạt động", f"{row['Time Diff between first and last (Mins)'] / 1440:,.1f} ngày")
    k2.metric("Tổng giao dịch ETH", f"{int(row[TX_ETH_COL]):,}")
    k3.metric("Tổng ETH đã nhận", f"{row['total ether received']:,.2f}")
    k4.metric("Giao dịch ERC20", f"{int(row[TX_ERC20_COL]):,}")

    st.write("")
    ok, msg = check_sufficient_data(row)
    if ok:
        show_result(predict_wallet(row, features), row)
    else:
        st.info(msg)

# ====================== CÁCH 2: NHẬP TAY ======================
with tab_manual:
    st.write("Nhập 6 thông số cơ bản của ví. Hệ thống tự động suy diễn các thông số liên quan và lấy trung vị ví bình thường cho các thuộc tính còn lại.")
    medians = normal_medians(df_train, features)

    with st.form("manual_form", border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            total_tx_in = st.number_input(
                "Tổng số giao dịch ETH (giao dịch)",
                min_value=0,
                value=int(round(float(medians[TX_ETH_COL]))),
                step=1,
            )
            days_in = st.number_input(
                "Tuổi hoạt động của ví (ngày)",
                min_value=0.0,
                value=round(float(medians["Time Diff between first and last (Mins)"]) / 1440.0, 1),
                step=1.0,
                format="%.1f",
            )
        with c2:
            sent_tx_in = st.number_input(
                "Số giao dịch gửi đi (giao dịch)",
                min_value=0,
                value=int(round(float(medians["Sent tnx"]))),
                step=1,
            )
            eth_rec_in = st.number_input(
                "Tổng ETH đã nhận (ETH)",
                min_value=0.0,
                value=round(float(medians["total ether received"]), 2),
                step=1.0,
                format="%.2f",
            )
        with c3:
            uniq_rec_in = st.number_input(
                "Số ví khác nhau đã gửi tiền đến (ví)",
                min_value=0,
                value=int(round(float(medians["Unique Received From Addresses"]))),
                step=1,
            )
            erc20_sent_in = st.number_input(
                "Tổng token ERC20 đã gửi (token)",
                min_value=0.0,
                value=round(float(medians["ERC20 total ether sent"]), 2),
                step=1.0,
                format="%.2f",
            )

        submitted = st.form_submit_button("Phân tích rủi ro & Đề xuất", type="primary", width="stretch")

    if submitted:
        entered = {
            "Time Diff between first and last (Mins)": float(days_in) * 1440.0,
            TX_ETH_COL: float(total_tx_in),
            "Sent tnx": float(sent_tx_in),
            "Unique Received From Addresses": float(uniq_rec_in),
            "total ether received": float(eth_rec_in),
            "ERC20 total ether sent": float(erc20_sent_in),
        }
        row = build_manual_wallet(medians, entered)
        ok, msg = check_sufficient_data(row)
        if ok:
            show_result(predict_wallet(row, features), row)
        else:
            st.info(msg)

# ====================== CÁCH 3: FILE CSV ======================
with tab_csv:
    st.write("Tải lên file CSV chứa nhiều ví để dự đoán cùng lúc. Chưa có file? Tải file mẫu bên dưới để xem định dạng.")
    sample = df_test[[ID_COL] + features].head(10)
    st.download_button(
        "Tải file CSV mẫu (10 ví)",
        sample.to_csv(index=False).encode("utf-8"),
        "mau_vi_ethereum.csv",
        "text/csv",
        icon=":material/download:",
    )

    file = st.file_uploader("Tải lên file CSV cần dự đoán", type=["csv"])
    if file is not None:
        up = pd.read_csv(file)
        up.columns = up.columns.str.strip()
        missing = [c for c in features if c not in up.columns]
        if missing:
            st.error(f"File thiếu {len(missing)} cột bắt buộc, ví dụ: " + ", ".join(missing[:5]))
        else:
            probs = predict_many(up, features)
            levels = []
            recommendations = []
            for i, p in enumerate(probs):
                row_i = up.iloc[i]
                ok, _ = check_sufficient_data(row_i)
                if ok:
                    lv = classify_risk(p)[0]
                    levels.append(lv)
                    recommendations.append(summarize_case_for_csv(row_i, lv, float(p)))
                else:
                    levels.append("Không đủ dữ liệu")
                    recommendations.append("Ví chưa có giao dịch ETH và ERC20 để đánh giá.")

            out = pd.DataFrame({
                "Địa chỉ ví": up[ID_COL] if ID_COL in up.columns else np.arange(1, len(up) + 1),
                "Xác suất gian lận (%)": np.round(probs * 100, 2),
                "Mức rủi ro": levels,
                "Khuyến nghị cụ thể": recommendations,
            })
            st.success(f"Đã phân tích {len(out)} ví bằng XGBoost.")

            n_high = sum(lv.startswith("CAO") for lv in levels)
            n_mid = sum(lv == "CẦN XEM XÉT" for lv in levels)
            n_low = sum(lv == "THẤP" for lv in levels)
            m1, m2, m3 = st.columns(3)
            m1.metric("Rủi ro cao", n_high)
            m2.metric("Cần xem xét", n_mid)
            m3.metric("Rủi ro thấp", n_low)

            st.dataframe(
                out,
                width="stretch",
                hide_index=True,
                column_config={
                    "Xác suất gian lận (%)": st.column_config.ProgressColumn(
                        "Xác suất gian lận (%)", min_value=0, max_value=100, format="%.1f%%"
                    ),
                    "Khuyến nghị cụ thể": st.column_config.TextColumn("Khuyến nghị cụ thể", width="large"),
                },
            )
            st.download_button(
                "Tải kết quả (CSV)",
                out.to_csv(index=False).encode("utf-8-sig"),
                "ket_qua.csv",
                "text/csv",
                type="primary",
                icon=":material/download:",
            )

st.write("")
with st.expander("Lời khuyên an toàn chung khi giao dịch Ethereum"):
    with st.container(border=True):
        for idx, (title, detail) in enumerate(GENERAL_TIPS, start=1):
            if idx > 1:
                st.divider()
            st.markdown(f"**{idx}. {title}**")
            st.write(detail)