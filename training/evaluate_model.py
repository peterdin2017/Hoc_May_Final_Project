"""
evaluate_model.py - Đánh giá 3 mô hình trên tập Test bằng:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC và ma trận nhầm lẫn.
(Đây là bài toán PHÂN LOẠI nên dùng các chỉ số này thay cho MAE, MSE, RMSE, R².)

Kết quả được lưu vào models/metrics.csv và models/test_predictions.csv
để các trang của ứng dụng đọc lại, không cần tính lại.

Chạy:  python training/evaluate_model.py   (sau khi đã chạy train_model.py)
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from preprocessing import (
    ID_COL, TARGET_COL, load_clean_data, get_feature_cols, split_data,
)
from train_model import MODELS_DIR

THRESHOLD = 0.5  # xác suất >= 0.5 thì xem là ví gian lận
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def main():
    df = load_clean_data()
    feature_cols = get_feature_cols(df)
    _, df_test = split_data(df)
    X_test = df_test[feature_cols]
    y_test = df_test[TARGET_COL]

    rows = []
    preds = pd.DataFrame({ID_COL: df_test[ID_COL].values, "y_true": y_test.values})

    for file_name, title in MODEL_NAMES.items():
        model = joblib.load(os.path.join(MODELS_DIR, f"{file_name}.pkl"))
        prob = model.predict_proba(X_test)[:, 1]  # xác suất là ví gian lận
        pred = (prob >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        rows.append({
            "Model": title,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, prob),
            "PR_AUC": average_precision_score(y_test, prob),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
        preds[f"prob_{title}"] = prob

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False)
    preds.to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)

    print(f"KẾT QUẢ TRÊN TẬP TEST ({len(df_test)} ví, ngưỡng {THRESHOLD})")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()