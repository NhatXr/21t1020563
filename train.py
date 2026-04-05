"""
TRAIN.PY - HỌC TẬP MÔ HÌNH SVM
===============================
Script huấn luyện mô hình phân loại bình luận tiếng Việt

Input: data/dataset_1000.xlsx
Output: 
  - data/dataset_1000.csv (cho app.py)
  - models/svm_model.pkl
  - models/vectorizer.pkl

Cách chạy:
    python train.py
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 HỌC TẬP MÔ HÌNH SVM - PHÂN LOẠI BÌNH LUẬN TIẾNG VIỆT")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TẢI DỮ LIỆU
# ══════════════════════════════════════════════════════════════════════════════
print("\n[BƯỚC 1] Tải dữ liệu...")

try:
    df = pd.read_excel("data/dataset_1000.xlsx")
    print(f"✅ Tải từ Excel: {len(df)} bình luận")
except FileNotFoundError:
    print("❌ Không tìm thấy data/dataset_1000.xlsx")
    print("   Vui lòng tạo thư mục data/ và đặt file vào")
    exit(1)

# Lưu thành CSV cho app.py
print("   📝 Lưu thành CSV...")
df.to_csv("data/dataset_1000.csv", index=False, encoding='utf-8')
print("   ✅ data/dataset_1000.csv")

print(f"\n   📊 Phân phối nhãn:")
for label, count in df['label'].value_counts().sort_index().items():
    label_name = {0: "Bình thường", 1: "Tiêu cực"}.get(label, f"Label {label}")
    print(f"      {label_name:15s}: {count:4d} ({count/len(df)*100:5.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: TIỀN XỬ LÝ (GIỐNG HẾT APP.PY)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[BƯỚC 2] Tiền xử lý dữ liệu...")

def preprocess_text(text):
    """Tiền xử lý - GIỐNG HỆT hàm trong app.py"""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(
        r"[^\w\sàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]",
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = df.dropna(subset=['comment', 'label'])
df['text_clean'] = df['comment'].apply(preprocess_text)
print(f"✅ Tiền xử lý xong: {len(df)} bình luận")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: CHIA TRAIN/TEST
# ══════════════════════════════════════════════════════════════════════════════
print("\n[BƯỚC 3] Chia train/test (80-20)...")

X = df['text_clean']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Train: {len(X_train):4d} | Test: {len(X_test):4d}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: TF-IDF VECTORIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[BƯỚC 4] TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
    min_df=2,
    max_df=0.8,
    lowercase=True
)

print("   ⏳ Fitting...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✅ Features: {X_train_vec.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: HUẤN LUYỆN SVM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[BƯỚC 5] Huấn luyện SVC RBF Kernel...")

model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42,
    max_iter=2000,
    verbose=0
)

print("   ⏳ Training...")
model.fit(X_train_vec, y_train)
print("✅ Huấn luyện xong!")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 6: ĐÁNH GIÁ
# ══════════════════════════════════════════════════════════════════════════════
print("\n[BƯỚC 6] Đánh giá mô hình...")

y_pred_train = model.predict(X_train_vec)
y_pred_test = model.predict(X_test_vec)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, zero_division=0)
recall = recall_score(y_test, y_pred_test, zero_division=0)
f1 = f1_score(y_test, y_pred_test, zero_division=0)

print("\n" + "=" * 80)
print("📊 KẾT QUẢ")
print("=" * 80)
print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:6.2f}%)")
print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:6.2f}%)")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-Score:          {f1:.4f}")
print("=" * 80)

print("\n📋 Classification Report:")
print(classification_report(
    y_test, y_pred_test,
    target_names=["Bình thường (0)", "Tiêu cực (1)"],
    digits=4,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred_test)
print("\n📊 Confusion Matrix:")
print(f"                Predicted")
print(f"              Bình thường  Tiêu cực")
print(f"Actual Bình thường    {cm[0,0]:3d}         {cm[0,1]:3d}")
print(f"       Tiêu cực       {cm[1,0]:3d}         {cm[1,1]:3d}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 7: LƯU MÔ HÌNH
# ══════════════════════════════════════════════════════════════════════════════
print("\n[BƯỚC 7] Lưu mô hình...")

os.makedirs("models", exist_ok=True)

print("   📁 Lưu SVM model...")
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("   ✅ models/svm_model.pkl")

print("   📁 Lưu TF-IDF vectorizer...")
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("   ✅ models/vectorizer.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# HOÀN TẤT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("🎉 HOÀN THÀNH!")
print("=" * 80)
print(f"\n✅ Mô hình được lưu tại: models/")
print(f"✅ CSV được lưu tại: data/dataset_1000.csv")
print(f"\n📊 Thống kê:")
print(f"   • Accuracy: {test_acc*100:.2f}%")
print(f"   • F1-Score: {f1:.4f}")
print(f"   • Features: {X_train_vec.shape[1]}")
print(f"\n🚀 Bước tiếp theo:")
print(f"   streamlit run app.py")
print("=" * 80)