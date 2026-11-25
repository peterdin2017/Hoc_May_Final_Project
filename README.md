# Phát hiện ví Ethereum gian lận bằng học máy

Đồ án môn Học máy. Ứng dụng web (Streamlit) dùng các mô hình học máy để dự đoán một ví Ethereum có dấu hiệu gian lận hay không, kèm khuyến nghị an toàn cho từng ví.

## 1. Giới thiệu

- **Dữ liệu:** Ethereum Fraud Detection (Kaggle). Dữ liệu thô có 9.841 ví, sau khi làm sạch còn 9.288 ví. Cột nhãn `FLAG` (1 = gian lận, 0 = bình thường).
- **Mô hình:** Logistic Regression, Random Forest, XGBoost.
- **Xử lý mất cân bằng lớp:** dùng `class_weight="balanced"` (Logistic Regression, Random Forest) và `scale_pos_weight` (XGBoost) để mô hình chú ý hơn tới lớp gian lận.
- **Chia dữ liệu:** 80% Train, 20% Test (chia phân tầng theo nhãn, `random_state=42`).
- **Ngưỡng phân loại:** xác suất gian lận >= 0.5 thì kết luận là ví gian lận; dưới 0.3 là rủi ro thấp.

### Kết quả trên tập Test

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8611 | 0.5717 | 0.8792 | 0.6929 | 0.9412 |
| Random Forest | 0.9553 | 0.9526 | 0.7885 | 0.8628 | 0.9861 |
| XGBoost | 0.9629 | 0.9043 | 0.8852 | 0.8947 | 0.9880 |

## 2. Cấu trúc thư mục

```
Hoc_May_Final_Project/
├── .streamlit/
│   └── config.toml              # Cấu hình giao diện Streamlit
├── app.py                       # Điểm vào của ứng dụng (trang chủ)
├── requirements.txt             # Danh sách thư viện cần cài
│
├── data/
│   ├── raw_ethereum_tx.csv      # Dữ liệu thô gốc
│   └── silver_ml_ready.csv      # Dữ liệu đã làm sạch, dùng để huấn luyện
│
├── notebooks/                   # Notebook thử nghiệm
│
├── training/                    # Code huấn luyện và đánh giá
│   ├── preprocessing.py         # Tiền xử lý, chia Train/Test
│   ├── train_model.py           # Huấn luyện 3 mô hình
│   ├── evaluate_model.py        # Đánh giá mô hình trên tập Test
│   └── feature_analysis.py      # Phân tích độ quan trọng của đặc trưng
│
├── models/                      # Mô hình đã huấn luyện và kết quả
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── metrics.csv
│   ├── test_predictions.csv
│   └── feature_importance.csv
│
├── src/                         # Hàm xử lý dùng chung
│   ├── prediction.py            # Đọc dữ liệu/mô hình, dự đoán
│   ├── recommendation.py        # Khuyến nghị an toàn cho từng ví
│   ├── visualization.py         # Các hàm vẽ biểu đồ
│   └── ui.py                    # CSS và thành phần giao diện
│
└── pages/                       # Các trang của ứng dụng
    ├── 1_Dashboard.py
    ├── 2_Data_Analysis.py
    ├── 3_Prediction.py
    └── 4_Model_Comparison.py
```

## 3. Yêu cầu hệ thống

- Python 3.10 trở lên
- Windows / macOS / Linux

## 4. Hướng dẫn cài đặt

**Bước 1:** Tải mã nguồn về máy

```bash
git clone https://github.com/peterdin2017/Hoc_May_Final_Project.git
cd Hoc_May_Final_Project
```

**Bước 2:** Tạo môi trường ảo (khuyến khích)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**Bước 3:** Cài thư viện

```bash
pip install -r requirements.txt
```

## 5. Hướng dẫn chạy ứng dụng

Đứng ở thư mục gốc của dự án và chạy:

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở địa chỉ `http://localhost:8501`. Nếu không tự mở, bạn copy địa chỉ này dán vào trình duyệt. Để tắt ứng dụng, quay lại terminal và nhấn `Ctrl + C`.

Các mô hình đã được huấn luyện sẵn trong thư mục `models/`, nên bạn chạy được ứng dụng ngay mà không cần huấn luyện lại.

## 6. Hướng dẫn sử dụng các trang

| Trang | Chức năng |
|---|---|
| Trang chủ (`app.py`) | Giới thiệu và điều hướng tới các trang |
| 1. Dashboard | Xem tổng quan dữ liệu và kết quả mô hình |
| 2. Data Analysis | Phân tích dữ liệu và trực quan hóa (biểu đồ phân phối, tương quan) |
| 3. Prediction | Dự đoán một ví có gian lận hay không, kèm khuyến nghị |
| 4. Model Comparison | So sánh các mô hình qua Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC |

### Cách dùng trang Prediction

Trang có 3 cách nhập dữ liệu:

1. **Chọn ví mẫu** trong tập Test để thử nhanh.
2. **Nhập tay** 6 thông số của ví cần kiểm tra.
3. **Tải file CSV** để dự đoán nhiều ví cùng lúc.

Sau khi nhập, chọn mô hình và bấm dự đoán. Kết quả gồm xác suất gian lận, mức rủi ro (thấp / trung bình / cao) và các đề xuất giải pháp an toàn.

## 7. Huấn luyện lại mô hình (tùy chọn)

Chạy lần lượt các lệnh sau từ thư mục gốc:

```bash
python training/train_model.py        # 1. Huấn luyện, lưu mô hình vào models/
python training/evaluate_model.py     # 2. Đánh giá, lưu metrics.csv và test_predictions.csv
python training/feature_analysis.py   # 3. Phân tích đặc trưng, lưu feature_importance.csv
```

## 8. Lỗi thường gặp

- **`ModuleNotFoundError`**: chưa cài thư viện, chạy lại `pip install -r requirements.txt`.
- **Lệnh `streamlit` không được nhận**: chưa kích hoạt môi trường ảo, hoặc chạy `python -m streamlit run app.py`.
- **Không tìm thấy file mô hình hoặc dữ liệu**: kiểm tra bạn đang đứng đúng thư mục gốc của dự án khi chạy lệnh.
- **Lỗi khi nạp file `.pkl`**: phiên bản `scikit-learn` / `xgboost` khác với lúc huấn luyện, hãy cài đúng phiên bản trong `requirements.txt` hoặc chạy lại `train_model.py`.

## 9. Tác giả

- Nhóm: <số nhóm>
- Thành viên: <tên các thành viên>
- Giảng viên hướng dẫn: <tên giảng viên>
- Trường: Đại học Công Thương TP.HCM
