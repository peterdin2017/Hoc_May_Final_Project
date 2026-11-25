"""
prediction.py - Đọc dữ liệu và mô hình đã huấn luyện, chuẩn bị đầu vào, thực hiện dự đoán.
Được dùng chung cho app.py và các trang trong pages/.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Thêm thư mục training/ vào đường dẫn để import preprocessing
# (joblib cần class SignedLog1pTransformer khi nạp file .pkl)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "training"))

from preprocessing import (  # noqa: E402
    ID_COL, TARGET_COL, TX_ETH_COL, TX_ERC20_COL,
    load_clean_data, get_feature_cols, split_data, check_sufficient_data,
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Tên file .pkl  ->  tên hiển thị
MODEL_FILES = {
    "XGBoost": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}
MODEL_NAMES = list(MODEL_FILES)

THRESHOLD = 0.5  # xác suất >= 0.5 thì kết luận là ví gian lận
LOW_LIMIT = 0.3  # dưới 0.3 là rủi ro thấp


# ------------------------------------------------------------------
# NẠP DỮ LIỆU VÀ MÔ HÌNH (cache để không đọc lại mỗi lần bấm nút)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    """Trả về (df đầy đủ, df_train, df_test, danh sách cột đặc trưng)."""
    df = load_clean_data()
    df_train, df_test = split_data(df)
    return df, df_train, df_test, get_feature_cols(df)


@st.cache_resource
def load_models():
    """Trả về dict {tên mô hình: mô hình đã huấn luyện}."""
    try:
        return _read_models()
    except Exception:
        # File .pkl lưu bằng phiên bản thư viện khác (hoặc bị hỏng) -> huấn luyện lại ngay trên máy này
        with st.spinner("Mô hình không tương thích với phiên bản thư viện trên máy, đang huấn luyện lại (khoảng 1 phút)..."):
            import train_model
            train_model.main()
        return _read_models()


def _read_models():
    return {name: joblib.load(os.path.join(MODELS_DIR, f)) for name, f in MODEL_FILES.items()}


@st.cache_data
def load_metrics():
    return pd.read_csv(os.path.join(MODELS_DIR, "metrics.csv"))


@st.cache_data
def load_test_predictions():
    return pd.read_csv(os.path.join(MODELS_DIR, "test_predictions.csv"))


@st.cache_data
def load_importance():
    return pd.read_csv(os.path.join(MODELS_DIR, "feature_importance.csv"))


# ------------------------------------------------------------------
# DỰ ĐOÁN
# ------------------------------------------------------------------
def predict_wallet(row, feature_cols):
    """
    row: một ví (dict hoặc Series) có đủ các cột đặc trưng.
    Trả về dict {tên mô hình: xác suất gian lận}.
    """
    X = pd.DataFrame([[float(row[c]) for c in feature_cols]], columns=feature_cols)
    models = load_models()
    return {name: float(m.predict_proba(X)[0, 1]) for name, m in models.items()}


def predict_many(df, feature_cols, model_name="XGBoost"):
    """Dự đoán xác suất gian lận cho nhiều ví cùng lúc (1 mô hình)."""
    return load_models()[model_name].predict_proba(df[feature_cols].fillna(0))[:, 1]


def normal_medians(df_train, feature_cols):
    """Trung vị của các ví bình thường - dùng làm giá trị mặc định khi nhập tay."""
    return df_train[df_train[TARGET_COL] == 0][feature_cols].median().to_dict()


def build_manual_wallet(medians, entered):
    """
    Từ 6 thông số người dùng nhập tay, tự động tính lại các cột có quan hệ toán học trực tiếp
    (số giao dịch nhận, trung bình ETH mỗi lần nhận, khoảng cách thời gian giữa các giao dịch,
    số giao dịch ERC20...) rồi ghép với trung vị của ví bình thường cho đủ 32 thuộc tính.
    """
    row = dict(medians)
    row.update(entered)

    time_mins = max(float(row["Time Diff between first and last (Mins)"]), 0.0)
    sent_tx = max(int(round(float(row["Sent tnx"]))), 0)
    uniq_rec = max(int(round(float(row["Unique Received From Addresses"]))), 0)
    eth_rec = max(float(row["total ether received"]), 0.0)
    erc20_sent = max(float(row["ERC20 total ether sent"]), 0.0)

    # Đảm bảo Tổng giao dịch ETH >= Số giao dịch gửi + Số giao dịch nhận tối thiểu
    min_rec_tx = max(uniq_rec, 1 if eth_rec > 0 else 0)
    total_tx = max(int(round(float(row[TX_ETH_COL]))), sent_tx + min_rec_tx)
    rec_tx = max(total_tx - sent_tx, min_rec_tx)
    total_tx = sent_tx + rec_tx

    row[TX_ETH_COL] = float(total_tx)
    row["Sent tnx"] = float(sent_tx)
    row["Received Tnx"] = float(rec_tx)
    row["Unique Received From Addresses"] = float(uniq_rec)

    # Đồng bộ các cột chiều nhận ETH (avg val received là thuộc tính quan trọng số 1)
    if rec_tx > 0 and eth_rec > 0:
        avg_rec = eth_rec / rec_tx
        row["avg val received"] = avg_rec
        row["min value received"] = min(float(medians["min value received"]), avg_rec)
        row["max value received"] = max(avg_rec, min(eth_rec, float(medians["max value received"])))
        row["Avg min between received tnx"] = time_mins / rec_tx if rec_tx > 1 else time_mins
    else:
        row["avg val received"] = 0.0
        row["min value received"] = 0.0
        row["max value received"] = 0.0
        row["Avg min between received tnx"] = 0.0

    # Đồng bộ các cột chiều gửi ETH
    if sent_tx > 0:
        row["Avg min between sent tnx"] = time_mins / sent_tx if sent_tx > 1 else time_mins
        row["Unique Sent To Addresses"] = float(min(max(int(medians["Unique Sent To Addresses"]), 1), sent_tx))
        est_sent = min(float(medians["total Ether sent"]), eth_rec if eth_rec > 0 else float(medians["total Ether sent"]))
        row["total Ether sent"] = est_sent
        row["avg val sent"] = est_sent / sent_tx
        row["min val sent"] = min(float(medians["min val sent"]), row["avg val sent"])
        row["max val sent"] = max(row["avg val sent"], min(est_sent, float(medians["max val sent"])))
    else:
        row["Avg min between sent tnx"] = 0.0
        row["Unique Sent To Addresses"] = 0.0
        row["total Ether sent"] = 0.0
        row["avg val sent"] = 0.0
        row["min val sent"] = 0.0
        row["max val sent"] = 0.0

    row["total ether balance"] = eth_rec - row["total Ether sent"]

    # Đồng bộ các cột Token ERC20 khi người dùng nhập lượng token ERC20 gửi đi
    if erc20_sent > 0:
        row[TX_ERC20_COL] = max(float(medians[TX_ERC20_COL]), 1.0)
        row["ERC20 avg val sent"] = erc20_sent
        row["ERC20 max val sent"] = erc20_sent
        row["ERC20 min val sent"] = erc20_sent
        row["ERC20 uniq sent addr"] = max(float(medians["ERC20 uniq sent addr"]), 1.0)
        row["ERC20 uniq sent token name"] = max(float(medians["ERC20 uniq sent token name"]), 1.0)

    return row


def classify_risk(prob):
    """Trả về (mức rủi ro, loại hộp thông báo) theo xác suất."""
    if prob < LOW_LIMIT:
        return "THẤP", "success"
    elif prob < THRESHOLD:
        return "CẦN XEM XÉT", "warning"
    else:
        return "CAO (nghi ngờ gian lận)", "error"