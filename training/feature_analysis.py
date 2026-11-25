"""
feature_analysis.py - Phân tích mức độ đóng góp của từng đặc trưng vào kết quả dự đoán
(ví dụ: tuổi ví, số giao dịch, lượng ETH nhận, token ERC20...).

Dùng 3 cách đo, mỗi cách đã chuẩn hóa để tổng bằng 1:
    - Logistic Regression: độ lớn của hệ số |coef|
    - Random Forest:       feature_importances_
    - XGBoost:             feature_importances_
Kết quả lưu ở models/feature_importance.csv.

Chạy:  python training/feature_analysis.py   (sau khi đã chạy train_model.py)
"""

import os
import joblib
import pandas as pd

from preprocessing import load_clean_data, get_feature_cols
from train_model import MODELS_DIR


def main():
    feature_cols = get_feature_cols(load_clean_data())
    result = pd.DataFrame({"feature": feature_cols})

    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))

    values = {
        "Logistic Regression": abs(lr.named_steps["clf"].coef_[0]),
        "Random Forest": rf.named_steps["clf"].feature_importances_,
        "XGBoost": xgb.named_steps["clf"].feature_importances_,
    }
    for name, v in values.items():
        result[name] = v / v.sum()  # chuẩn hóa: tổng = 1

    result["Trung bình"] = result[list(values)].mean(axis=1)
    result = result.sort_values("Trung bình", ascending=False).round(5)
    result.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)

    print("10 ĐẶC TRƯNG ĐÓNG GÓP NHIỀU NHẤT (trung bình 3 mô hình):")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()