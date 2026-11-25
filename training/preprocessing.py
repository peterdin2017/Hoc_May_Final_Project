"""
preprocessing.py - Các hàm tiền xử lý dữ liệu dùng chính thức khi huấn luyện và khi chạy ứng dụng.

Luồng xử lý:  đọc data/silver_ml_ready.csv (đã sạch)  ->  chia Train / Test
(Các hàm load_raw_data / clean_data giữ lại để đối chiếu với dữ liệu thô.)
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

# Đường dẫn: thư mục gốc của đồ án là thư mục cha của training/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw_ethereum_tx.csv")        # dữ liệu thô (bronze)
SILVER_PATH = os.path.join(BASE_DIR, "data", "silver_ml_ready.csv")      # dữ liệu đã sạch, sẵn sàng cho ML (silver) <- DÙNG CHÍNH

ID_COL = "Address"  # địa chỉ ví (chỉ để tra cứu, KHÔNG đưa vào mô hình)
INDEX_COL = "source_index"  # số thứ tự dòng trong file silver (KHÔNG phải đặc trưng, bỏ khi nạp)
TARGET_COL = "FLAG"  # nhãn: 0 = bình thường, 1 = gian lận
TX_ETH_COL = "total transactions (including tnx to create contract"
TX_ERC20_COL = "Total ERC20 tnxs"

RANDOM_STATE = 42

# 6 cột gần như toàn số 0 (>= 99,7%), không có ích cho mô hình
NEAR_CONSTANT_COLS = [
    "min value sent to contract",
    "max val sent to contract",
    "avg value sent to contract",
    "total ether sent contracts",
    "ERC20 total Ether sent contract",
    "ERC20 uniq sent addr.1",
]


class SignedLog1pTransformer(BaseEstimator, TransformerMixin):
    """
    Biến đổi sign(x) * log(1 + |x|).
    Dữ liệu có những giá trị cực lớn (đến 10^12), log giúp "nén" chúng lại.
    Giữ nguyên dấu âm (cột 'total ether balance' có giá trị âm).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))


# ------------------------------------------------------------------
# 1. ĐỌC DỮ LIỆU THÔ
# ------------------------------------------------------------------
def load_raw_data(path=RAW_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # một số tên cột bị thừa khoảng trắng
    return df


# ------------------------------------------------------------------
# 2. LÀM SẠCH
# ------------------------------------------------------------------
def clean_data(df):
    """Trả về DataFrame gồm: Address, FLAG và 32 cột đặc trưng số."""
    df = df.copy()

    # Bỏ cột chỉ mục thừa
    df = df.drop(columns=[c for c in ["Unnamed: 0", "Index"] if c in df.columns])

    # Bỏ cột hằng số (chỉ có 1 giá trị duy nhất)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const_cols)

    # Bỏ cột chữ (tên token...), nhưng GIỮ cột Address
    text_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c != ID_COL]
    df = df.drop(columns=text_cols)

    # Điền 0 vào ô trống và xóa dòng trùng lặp
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    df[num_cols] = df[num_cols].fillna(0)
    df = df.drop_duplicates(subset=num_cols, keep="first").reset_index(drop=True)

    # Bỏ 6 cột gần hằng số
    df = df.drop(columns=[c for c in NEAR_CONSTANT_COLS if c in df.columns])

    # Sắp xếp cột: Address, FLAG, rồi các đặc trưng
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]
    return df[[ID_COL, TARGET_COL] + feature_cols]


def get_feature_cols(df):
    """Danh sách tên cột đặc trưng (không gồm Address và FLAG)."""
    return [c for c in df.columns if c not in [ID_COL, TARGET_COL]]


def load_silver_data(path=SILVER_PATH):
    """
    Đọc file silver_ml_ready.csv (đã làm sạch sẵn): Address, FLAG và 32 đặc trưng số.
    Bỏ cột chỉ mục `source_index` để nó không lọt vào mô hình.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in [INDEX_COL] if c in df.columns])

    # Kiểm tra an toàn: file phải có nhãn, địa chỉ và toàn bộ đặc trưng là số
    assert ID_COL in df.columns and TARGET_COL in df.columns, "Thiếu cột Address hoặc FLAG"
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df[[ID_COL, TARGET_COL] + feature_cols]


def load_clean_data():
    """Dữ liệu dùng cho huấn luyện và ứng dụng: đọc từ file silver."""
    return load_silver_data()


# ------------------------------------------------------------------
# 3. CHIA TRAIN / TEST
# ------------------------------------------------------------------
def split_data(df, test_size=0.2):
    """
    Chia 80% Train / 20% Test, giữ nguyên tỷ lệ gian lận ở 2 tập (stratify).
    random_state cố định nên lần nào chạy cũng ra cùng một cách chia.
    """
    return train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )


def check_sufficient_data(row):
    """Ví chưa có giao dịch ETH lẫn ERC20 thì không đủ dữ liệu để dự đoán."""
    eth = float(row.get(TX_ETH_COL, 0))
    erc = float(row.get(TX_ERC20_COL, 0))
    if eth <= 0 and erc <= 0:
        return False, "Không đủ dữ liệu: ví chưa có giao dịch ETH lẫn token ERC20."
    return True, ""