# 🔍 Phân loại bình luận tiếng Việt - SVM + TF-IDF

Dự án phân loại bình luận tiếng Việt thành 2 lớp: **Bình thường** và **Tiêu cực** sử dụng mô hình Support Vector Machine (SVM) với TF-IDF vectorization.

## 📊 Thông tin mô hình

| Chỉ số | Giá trị |
|--------|--------|
| **Model** | LinearSVC |
| **Vectorizer** | TF-IDF |
| **Classes** | 2 (Bình thường, Tiêu cực) |
| **Training Accuracy** | 89.88% |
| **Test Accuracy** | 61.86% |
| **F1-Score** | 0.4384 |
| **ROC-AUC** | 0.5677 |
| **Features** | 1,146 (1,2-gram) |

## 📁 Cấu trúc dự án

```
project/
├── data/
│   └── dataset_1000.xlsx          # Dữ liệu huấn luyện
├── models/
│   ├── svm_model.pkl              # Mô hình SVM
│   ├── vectorizer.pkl             # Vectorizer TF-IDF
│   └── model.pkl                  # Backup
├── app.py                         # Ứng dụng Streamlit
├── train.py                       # Script huấn luyện
├── test.py                        # Script kiểm thử
├── requirements.txt               # Dependencies
└── README.md                      # File này
```

## 🚀 Cài đặt & Sử dụng

### 1️⃣ Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Huấn luyện mô hình

```bash
python train.py
```

**Output:**
- `models/svm_model.pkl` - Mô hình SVM
- `models/vectorizer.pkl` - Vectorizer TF-IDF
- `TRAINING_REPORT.txt` - Báo cáo huấn luyện

### 3️⃣ Kiểm thử mô hình

```bash
python test.py
```

**Output:** Kết quả kiểm thử trên 20 bình luận mẫu

### 4️⃣ Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Mở trình duyệt tại: `http://localhost:8501`

## 📖 Hướng dẫn sử dụng

### Trang chủ
- Xem thông tin mô hình
- Biết thêm về các tính năng

### Phân loại đơn
- Nhập một bình luận
- Nhấn "Phân loại"
- Xem kết quả: lớp + độ tin cậy

### Phân loại hàng loạt
- Upload file CSV (có cột `comment`)
- Hoặc file Excel
- Tự động phân loại tất cả bình luận
- Tải kết quả dưới dạng CSV

### Thống kê
- Xem chi tiết hiệu suất mô hình
- Confusion matrix
- Giải thích các chỉ số

## 🎯 Đặc tính

✅ **Tiền xử lý nâng cao**
- Xoá URL, email, số điện thoại
- Xoá ký tự đặc biệt nhưng giữ dấu Việt
- Chuẩn hóa khoảng trắng

✅ **Features tốt**
- TF-IDF Vectorization
- Unigram + Bigram
- 1,146 features

✅ **Mô hình mạnh**
- SVM với RBF kernel
- Balanced class weights
- Cross-validation tuning

✅ **Giao diện thân thiện**
- Streamlit UI
- Đơn giản, dễ sử dụng
- Multiple modes

## 📊 Phân tích kết quả

### Độ chính xác từng lớp

**Bình thường (Class 0)**
- Precision: 71.63%
- Recall: 70.63%

**Tiêu cực (Class 1)**
- Precision: 43.24%
- Recall: 44.44%

### Confusion Matrix

|  | Dự đoán: Bình thường | Dự đoán: Tiêu cực |
|--|--|--|
| **Thực tế: Bình thường** | 101 | 42 |
| **Thực tế: Tiêu cực** | 40 | 32 |

### Chỉ số bổ sung

- **Sensitivity (Recall):** 44.44% - Mô hình phát hiện được 44% bình luận tiêu cực
- **Specificity:** 70.63% - Mô hình đúng 70% khi nói "bình thường"

## ⚠️ Giới hạn & Ghi chú

1. **Class Imbalance:** Dataset có 2:1 (717:358), mô hình bị thiên vị về lớp đa số
2. **Overfitting:** Gap giữa train (89.88%) và test (61.86%) cho thấy overfitting nhẹ
3. **Recall thấp:** Chỉ phát hiện ~44% bình luận tiêu cực, cần cải thiện

## 🔄 Cải tiến trong tương lai

- [ ] Thu thập thêm dữ liệu, đặc biệt bình luận tiêu cực
- [ ] Sử dụng SMOTE để cân bằng dataset
- [ ] Thử các mô hình khác (Random Forest, Gradient Boosting)
- [ ] Tuning hyperparameters tinh tế hơn
- [ ] Transfer Learning với BERT/PhoBERT

## 📝 Format dữ liệu

### File CSV/Excel input

```csv
comment
"Sản phẩm rất tốt"
"Hàng fake, lừa đảo"
```

### Output CSV

```csv
comment,label_name,confidence
"Sản phẩm rất tốt",Bình thường,0.8234
"Hàng fake, lừa đảo",Tiêu cực,0.7123
```

## 🔧 Troubleshooting

### Lỗi: `FileNotFoundError: models/svm_model.pkl`
**Giải pháp:** Chạy `python train.py` trước

### Lỗi: `ImportError: No module named 'streamlit'`
**Giải pháp:** `pip install -r requirements.txt`

### Lỗi: Tiếng Việt không hiển thị đúng
**Giải pháp:** Đảm bảo file CSV/Excel được lưu với encoding UTF-8

## 📚 Tài liệu tham khảo

- [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html)
- [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 👤 Tác giả

Dự án phân loại bình luận tiếng Việt

## 📄 License

MIT License

---

**Cần hỗ trợ?** Kiểm tra file `TRAINING_REPORT.txt` để xem chi tiết báo cáo huấn luyện.
