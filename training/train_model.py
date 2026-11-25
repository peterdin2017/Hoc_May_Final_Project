"""
train_model.py - Xây dựng và huấn luyện 3 thuật toán:
    Logistic Regression, Random Forest, XGBoost (Gradient Boosting).
Mô hình sau khi huấn luyện được lưu vào thư mục models/ (file .pkl).

Chạy:  python training/train_model.py
"""

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import (
    BASE_DIR, TARGET_COL, RANDOM_STATE, SignedLog1pTransformer,
    load_clean_data, get_feature_cols, split_data,
)

MODELS_DIR = os.path.join(BASE_DIR, "models")


def build_models(ratio):
    """
    Tạo 3 mô hình. Mỗi mô hình là một Pipeline (chuỗi bước) gồm:
        log (nén số lớn)  ->  [chuẩn hóa]  ->  thuật toán
    Dữ liệu chỉ ~18% là ví gian lận (mất cân bằng) nên cho mô hình
    "chú ý" nhiều hơn tới lớp gian lận (class_weight / scale_pos_weight).
    ratio = số ví bình thường / số ví gian lận.
    """
    return {
        "logistic_regression": Pipeline([
            ("log", SignedLog1pTransformer()),
            ("scaler", StandardScaler()),  # Logistic Regression cần chuẩn hóa
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("log", SignedLog1pTransformer()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("log", SignedLog1pTransformer()),
            ("clf", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                  scale_pos_weight=ratio, eval_metric="logloss",
                                  random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def main():
    df = load_clean_data()
    feature_cols = get_feature_cols(df)
    df_train, df_test = split_data(df)

    X_train = df_train[feature_cols]
    y_train = df_train[TARGET_COL]
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {len(df_train)} ví | Test: {len(df_test)} ví | {len(feature_cols)} đặc trưng")

    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in build_models(ratio).items():
        print(f"Đang huấn luyện {name} ...")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
        print(f"  -> đã lưu models/{name}.pkl")


if __name__ == "__main__":
    main()