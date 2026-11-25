"""
visualization.py - Các hàm tạo biểu đồ (Plotly) và tên tiếng Việt của thuộc tính.
Mỗi hàm trả về một biểu đồ; trang gọi st.plotly_chart(biểu đồ) để hiển thị.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

COLORS = {"Bình thường": "#3B82A0", "Gian lận": "#D64545"}
MODEL_COLORS = ["#0F766E", "#3B82A0", "#E0A100", "#8B5CF6"]
INK = "#0B1F33"


def _style(fig, height=380, legend_bottom=True):
    """Áp dụng kiểu chung cho mọi biểu đồ: nền trắng, chữ gọn, lưới nhạt."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Be Vietnam Pro, Segoe UI, sans-serif", size=13),
        title=dict(font=dict(size=15), x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=MODEL_COLORS,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0) if legend_bottom else dict(),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig

# Cột thời gian trong dữ liệu gốc tính bằng PHÚT -> hiển thị bằng NGÀY
TIME_COLS = [
    "Time Diff between first and last (Mins)",
    "Avg min between received tnx",
    "Avg min between sent tnx",
]

# Tên tiếng Việt: cột gốc -> (tên hiển thị, đơn vị)
FEATURE_VI = {
    "Time Diff between first and last (Mins)": ("Tuổi hoạt động của ví", "ngày"),
    "Avg min between received tnx": ("TB bao lâu nhận tiền 1 lần", "ngày"),
    "Avg min between sent tnx": ("TB bao lâu gửi tiền 1 lần", "ngày"),
    "Sent tnx": ("Số giao dịch gửi đi", "giao dịch"),
    "Received Tnx": ("Số giao dịch nhận về", "giao dịch"),
    "Number of Created Contracts": ("Số hợp đồng đã tạo", "hợp đồng"),
    "Unique Received From Addresses": ("Số ví khác nhau đã gửi tiền đến", "ví"),
    "Unique Sent To Addresses": ("Số ví khác nhau đã nhận tiền từ ví này", "ví"),
    "total transactions (including tnx to create contract": ("Tổng số giao dịch ETH", "giao dịch"),
    "total ether received": ("Tổng ETH đã nhận", "ETH"),
    "total Ether sent": ("Tổng ETH đã gửi", "ETH"),
    "total ether balance": ("Chênh lệch ETH (nhận - gửi)", "ETH"),
    "min value received": ("ETH nhận nhỏ nhất 1 lần", "ETH"),
    "max value received": ("ETH nhận lớn nhất 1 lần", "ETH"),
    "avg val received": ("ETH nhận trung bình 1 lần", "ETH"),
    "min val sent": ("ETH gửi nhỏ nhất 1 lần", "ETH"),
    "max val sent": ("ETH gửi lớn nhất 1 lần", "ETH"),
    "avg val sent": ("ETH gửi trung bình 1 lần", "ETH"),
    "Total ERC20 tnxs": ("Tổng số giao dịch token ERC20", "giao dịch"),
    "ERC20 total Ether received": ("Tổng token ERC20 đã nhận", "token"),
    "ERC20 total ether sent": ("Tổng token ERC20 đã gửi", "token"),
    "ERC20 uniq sent addr": ("Số ví khác nhau được gửi token", "ví"),
    "ERC20 uniq rec addr": ("Số ví khác nhau đã gửi token đến", "ví"),
    "ERC20 uniq rec contract addr": ("Số hợp đồng token khác nhau đã nhận", "hợp đồng"),
    "ERC20 min val rec": ("Token nhận nhỏ nhất 1 lần", "token"),
    "ERC20 max val rec": ("Token nhận lớn nhất 1 lần", "token"),
    "ERC20 avg val rec": ("Token nhận trung bình 1 lần", "token"),
    "ERC20 min val sent": ("Token gửi nhỏ nhất 1 lần", "token"),
    "ERC20 max val sent": ("Token gửi lớn nhất 1 lần", "token"),
    "ERC20 avg val sent": ("Token gửi trung bình 1 lần", "token"),
    "ERC20 uniq sent token name": ("Số loại token khác nhau đã gửi", "loại"),
    "ERC20 uniq rec token name": ("Số loại token khác nhau đã nhận", "loại"),
}


def vi_name(col):
    """Tên tiếng Việt của thuộc tính."""
    return FEATURE_VI.get(col, (col, ""))[0]


def vi_label(col):
    """Tên tiếng Việt kèm đơn vị, ví dụ 'Tổng ETH đã nhận (ETH)'."""
    name, unit = FEATURE_VI.get(col, (col, ""))
    return f"{name} ({unit})" if unit else name


# ------------------------------------------------------------------
# PHÂN TÍCH DỮ LIỆU
# ------------------------------------------------------------------
def plot_label_pie(n_normal, n_fraud):
    data = pd.DataFrame({"Loại ví": ["Bình thường", "Gian lận"], "Số lượng": [n_normal, n_fraud]})
    fig = px.pie(data, names="Loại ví", values="Số lượng", hole=0.55, color="Loại ví",
                 color_discrete_map=COLORS, title="Tỷ lệ ví bình thường và ví gian lận")
    fig.update_traces(textinfo="percent", textfont_size=14)
    return _style(fig, 360)


def plot_histogram(values, labels):
    """Biểu đồ phân phối của 1 thuộc tính theo 2 nhóm ví (thang log vì dữ liệu lệch mạnh)."""
    v = np.asarray(values, dtype=float)
    d = pd.DataFrame({"Giá trị (log)": np.sign(v) * np.log1p(np.abs(v)), "Loại ví": labels})
    fig = px.histogram(d, x="Giá trị (log)", color="Loại ví", barmode="overlay", opacity=0.65,
                       nbins=40, color_discrete_map=COLORS, title="Phân phối (thang log)")
    return _style(fig, 360)


def plot_boxplot(values, labels):
    v = np.asarray(values, dtype=float)
    d = pd.DataFrame({"Giá trị (log)": np.sign(v) * np.log1p(np.abs(v)), "Loại ví": labels})
    fig = px.box(d, x="Loại ví", y="Giá trị (log)", color="Loại ví",
                 color_discrete_map=COLORS, title="Boxplot so sánh 2 nhóm")
    fig.update_layout(showlegend=False)
    return _style(fig, 360, legend_bottom=False)


def plot_corr_bar(corr):
    """corr: Series tương quan với nhãn (index = tên cột gốc)."""
    d = pd.DataFrame({"Thuộc tính": [vi_name(c) for c in corr.index], "Hệ số tương quan": corr.values})
    fig = px.bar(d.sort_values("Hệ số tương quan"), x="Hệ số tương quan", y="Thuộc tính",
                 orientation="h", title="Tương quan Pearson với nhãn gian lận")
    fig.update_traces(marker_color="#0F766E")
    fig.update_layout(yaxis_title="")
    return _style(fig, max(360, 28 * len(d) + 90), legend_bottom=False)


def plot_corr_heatmap(corr_matrix):
    m = corr_matrix.copy()
    m.index = [vi_name(c) for c in m.index]
    m.columns = [vi_name(c) for c in m.columns]
    fig = px.imshow(m, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    return _style(fig, 520, legend_bottom=False)


# ------------------------------------------------------------------
# MÔ HÌNH
# ------------------------------------------------------------------
def plot_metric_bar(metrics_df, metric_cols):
    """Biểu đồ cột so sánh các mô hình theo nhiều chỉ số."""
    long = metrics_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Chỉ số", value_name="Điểm")
    fig = px.bar(long, x="Chỉ số", y="Điểm", color="Model", barmode="group", text_auto=".3f")
    fig.update_layout(yaxis_range=[0, 1.08], yaxis_title="", xaxis_title="", legend_title_text="")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _style(fig, 400)


def plot_roc(y_true, probs):
    """probs: dict {tên mô hình: mảng xác suất}."""
    fig = go.Figure()
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc(fpr, tpr):.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Ngẫu nhiên"))
    fig.update_layout(title="Đường cong ROC", xaxis_title="Tỷ lệ báo nhầm (FPR)", yaxis_title="Tỷ lệ bắt đúng (Recall)")
    return _style(fig, 420)


def plot_pr(y_true, probs):
    fig = go.Figure()
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{name} (PR-AUC={average_precision_score(y_true, p):.3f})"))
    fig.update_layout(title="Đường cong Precision - Recall", xaxis_title="Recall", yaxis_title="Precision")
    return _style(fig, 420)


def plot_importance(imp_df, column, top=10):
    """Biểu đồ độ quan trọng của các thuộc tính. imp_df có cột 'feature' và cột `column`."""
    d = imp_df.sort_values(column, ascending=False).head(top).copy()
    d["Thuộc tính"] = d["feature"].apply(vi_name)
    d = d.sort_values(column)
    fig = px.bar(d, x=column, y="Thuộc tính", orientation="h")
    fig.update_traces(marker_color="#0F766E")
    fig.update_layout(xaxis_title="Độ quan trọng", yaxis_title="")
    return _style(fig, max(360, 34 * len(d) + 80), legend_bottom=False)


def plot_gauge(prob, threshold=0.5, low_limit=0.3):
    """Đồng hồ đo xác suất gian lận (0-100%) với 3 vùng: thấp / cần xem xét / cao."""
    value = float(min(max(prob, 0.0), 1.0)) * 100
    bar_color = "#16A34A" if prob < low_limit else ("#D97706" if prob < threshold else "#DC2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(size=44), valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 30, 50, 100], tickfont=dict(size=12)),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, low_limit * 100], color="rgba(22,163,74,.28)"),
                dict(range=[low_limit * 100, threshold * 100], color="rgba(217,119,6,.28)"),
                dict(range=[threshold * 100, 100], color="rgba(220,38,38,.28)"),
            ],
            threshold=dict(line=dict(color="#94A3B8", width=3), thickness=0.8, value=threshold * 100),
        ),
        title=dict(text="Xác suất gian lận (XGBoost)", font=dict(size=14)),
    ))
    fig.update_layout(
        height=290, margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Be Vietnam Pro, Segoe UI, sans-serif"),
    )
    return fig