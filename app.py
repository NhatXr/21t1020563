"""
APP.PY - STREAMLIT APPLICATION - PHIÊN BẢN NÂNG CẤP
====================================================
Phân loại bình luận tiếng Việt bằng SVM 
- 3 Pages: Giới thiệu, Phân loại, Đánh giá (nâng cấp)
- Page 3 có 5 tabs chi tiết
- Load model từ pickle files
- UI đẹp, không lỗi Streamlit cache

Cách chạy:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, cohen_kappa_score, hamming_loss
)
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân loại bình luận tiếng Việt - SVM ",
    page_icon="🛡️",
    layout="wide"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }
    
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1e2937;
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 10px 0 4px 0;
    }
    
    .student-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 16px;
        padding: 24px 28px;
        border: 1px solid #bae6fd;
        margin-bottom: 28px;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin: 32px 0 16px 0;
        border-left: 5px solid #3b82f6;
        padding-left: 14px;
    }
    
    .eval-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .result-box {
        border-radius: 14px;
        padding: 22px 26px;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.title("📌 Điều hướng")
page = st.sidebar.radio(
    "Chọn trang:",
    ["1. Giới thiệu & EDA", "2. Dự đoán bình luận", "3. Đánh giá mô hình"]
)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Đang tải mô hình…")
def load_model_from_file():
    """Load model và vectorizer từ file pickle"""
    try:
        model = pickle.load(open("models/svm_model.pkl", "rb"))
        vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy model: {e}")
        st.info("💡 Hãy chạy trước: python train.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi load model: {e}")
        st.stop()

model, vectorizer = load_model_from_file()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: GIỚI THIỆU & EDA
# ══════════════════════════════════════════════════════════════════════════════

if page == "1. Giới thiệu & EDA":

    st.markdown(
        '<div class="main-header">🛡️ Phân loại bình luận tiếng Việt bằng SVM cho các ứng dụng mạng xã hội</div>',
        unsafe_allow_html=True
    )
    st.markdown("**Hệ thống hỗ trợ kiểm duyệt nội dung tự động trên mạng xã hội**")

    st.markdown("""
    <div class="student-card">
        <div style="display:flex; align-items:center; gap:20px;">
            <div style="font-size:2.8rem;">👨‍🎓</div>
            <div>
                <h3 style="margin:0; color:#0f172a;">Nguyễn Thành Nhật</h3>
                <p style="margin:6px 0 0 0; color:#475569; font-size:1.05rem;">
                    MSSV: 21T1020563 &nbsp;|&nbsp; Đồ án: Phân loại bình luận tiếng Việt bằng SVM
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-title">🎯 1. Mục tiêu bài toán</div>', unsafe_allow_html=True)
    st.write(
        "Xây dựng mô hình **Support Vector Machine (SVM)** để tự động phân loại "
        "bình luận tiếng Việt, hỗ trợ kiểm duyệt nội dung trên các nền tảng "
        "mạng xã hội và diễn đàn trực tuyến."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **Bình luận bình thường**\n\nNội dung lành mạnh, không vi phạm")
    with col2:
        st.warning("⚠️ **Bình luận tiêu cực**\n\nThù địch, xúc phạm, miệt thị")

    st.divider()

    st.markdown('<div class="section-title">💡 2. Giá trị thực tiễn</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.success("🛡️ Tự động phát hiện bình luận độc hại theo thời gian thực")
        st.success("📉 Giảm đáng kể nhân lực kiểm duyệt thủ công")
        st.success("⚡ Tăng tốc độ xử lý khi khối lượng nội dung lớn")
    with c2:
        st.success("🤝 Bảo vệ cộng đồng khỏi nội dung thù địch và bạo lực")
        st.success("📊 Cung cấp báo cáo thống kê chất lượng bình luận")
        st.success("🔄 Dễ dàng tích hợp vào các hệ thống quản lý nội dung")

    st.divider()

    st.markdown('<div class="section-title">📊 3. Tổng quan dữ liệu</div>', unsafe_allow_html=True)

    try:
        df = pd.read_csv("data/dataset_1000.csv")
        
        st.subheader("📋 Dữ liệu mẫu (8 dòng đầu)")
        st.dataframe(df.head(8), use_container_width=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📝 Tổng bình luận", f"{len(df):,}")
        with col_b:
            st.metric("🏷️ Số lớp nhãn", df['label'].nunique())
        
        st.subheader("📈 Phân phối nhãn")
        if 'label' in df.columns:
            label_dist = df['label'].value_counts().sort_index()
            label_names = {0: "Bình thường", 1: "Tiêu cực"}
            
            labels = [label_names.get(i, f"Class {i}") for i in label_dist.index]
            colors = ['#22c55e', '#f59e0b'][:len(label_dist)]
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(labels, label_dist.values, color=colors, edgecolor='white', linewidth=1.2)
            
            for bar, val in zip(bars, label_dist.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val}\n({val/len(df)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='600', fontsize=10)
            
            ax.set_ylabel("Số lượng bình luận", fontsize=11)
            ax.set_title("Phân phối nhãn trong bộ dữ liệu", fontsize=13, fontweight='600')
            ax.spines[['top', 'right']].set_visible(False)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

    except FileNotFoundError:
        st.warning("⚠️ Không tìm thấy file: data/dataset_1000.csv")
        st.info("💡 Hãy chạy trước: python train.py")

    st.divider()

    st.markdown('<div class="section-title">🔄 4. Pipeline hệ thống</div>', unsafe_allow_html=True)
    
    steps = [
        ("Thu thập dữ liệu", "Bình luận từ Facebook, Tiktok và các nền tảng mạng xã hội khác."),
        ("Tiền xử lý văn bản", "Loại bỏ URL, email; chuẩn hoá dấu câu; lọc stopwords tiếng Việt."),
        ("Trích xuất đặc trưng", "TF-IDF Vectorizer chuyển văn bản thành vector số; n-gram (1,2); max_features=5000."),
        ("Huấn luyện mô hình SVM", "SVC kernel RBF; class_weight='balanced' để xử lý imbalance."),
        ("Đánh giá & tối ưu", "Accuracy, Precision, Recall, F1-score; Confusion Matrix; Cross-validation."),
        ("Triển khai dự đoán", "Xuất model (pickle); Streamlit UI; REST API (tùy chọn)."),
    ]
    
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div style="background: #f8fafc; border-radius: 10px; padding: 14px 18px; border: 1px solid #e2e8f0; margin-bottom: 10px; display: flex; align-items: flex-start; gap: 12px;">
            <div style="background: #3b82f6; color: white; border-radius: 50%; width: 28px; height: 28px; min-width: 28px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem;">{i}</div>
            <div><strong>{title}</strong><br>
            <span style="color:#475569;font-size:.95rem;">{desc}</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PHÂN LOẠI BÌNH LUẬN
# ══════════════════════════════════════════════════════════════════════════════

elif page == "2. Dự đoán bình luận":

    st.markdown(
        '<div class="main-header">🔍 Phân loại bình luận</div>',
        unsafe_allow_html=True
    )
    st.write("Nhập bình luận tiếng Việt — mô hình SVM sẽ tự động phân loại thành 2 lớp.")

    LABEL_CONFIG = {
        0: {
            "name": "✅ Bình thường",
            "bg_color": "#dcfce7",
            "text_color": "#166534",
            "icon": "🟢",
            "description": "Nội dung lành mạnh, không chứa yếu tố tiêu cực hay vi phạm",
        },
        1: {
            "name": "⚠️ Tiêu cực / Độc hại",
            "bg_color": "#fef9c3",
            "text_color": "#854d0e",
            "icon": "🟡",
            "description": "Nội dung chứa ngôn từ thù địch, xúc phạm, miệt thị, bi lăng nhăng",
        },
    }
    st.divider()

    default_text = st.session_state.get("example_text", "")
    comment_input = st.text_area(
        "✏️ Nhập bình luận cần phân loại:",
        value=default_text,
        height=130,
        placeholder="Nhập bình luận tiếng Việt vào đây…",
        key="comment_input"
    )

    char_count = len(comment_input)
    word_count = len(comment_input.split())
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.caption(f"📝 {char_count} ký tự | {word_count} từ")

    st.divider()

    if st.button("🚀 Phân loại bình luận", type="primary", use_container_width=True, key="classify_btn"):
        text = comment_input.strip()
        
        if not text or len(text) < 3:
            st.warning("⚠️ Vui lòng nhập ít nhất 3 ký tự.")
            st.stop()

        def preprocess(text_input):
            """Tiền xử lý văn bản"""
            text_input = str(text_input).lower().strip()
            text_input = re.sub(r"http\S+|www\S+|https\S+", " url ", text_input)
            text_input = re.sub(r"\S+@\S+", " email ", text_input)
            text_input = re.sub(
                r"[^\w\sàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]",
                " ",
                text_input
            )
            text_input = re.sub(r"\s+", " ", text_input).strip()
            return text_input

        text_clean = preprocess(text)

        try:
            X_vec = vectorizer.transform([text_clean])
            pred = int(model.predict(X_vec)[0])
            
            try:
                confidence = model.decision_function(X_vec)[0]
                confidence_score = float(1 / (1 + np.exp(-confidence)))
            except:
                confidence_score = None
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
            st.stop()

        config = LABEL_CONFIG.get(pred, LABEL_CONFIG[0])

        st.markdown(f"""
        <div style="
            background:{config['bg_color']};
            color:{config['text_color']};
            border: 3px solid {config['text_color']};
            border-radius: 16px;
            padding: 32px 28px;
            font-size: 1.6rem;
            font-weight: 700;
            text-align: center;
            margin: 24px 0 16px 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        ">
            {config['name']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**📌 Giải thích:**")
        st.info(config['description'])

        if confidence_score is not None:
            st.progress(confidence_score, text=f"🎯 Độ tin cậy: {confidence_score*100:.1f}%")

        if pred == 1:
            st.warning("⚠️ **CẢNH BÁO:** Nội dung này có yếu tố tiêu cực!")

        if "example_text" in st.session_state:
            st.session_state.pop("example_text", None)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ĐÁNH GIÁ MÔ HÌNH - PHIÊN BẢN NÂNG CẤP (5 TABS)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "3. Đánh giá mô hình":

    st.markdown("""
    <div class="eval-header">
        <h1 style="margin: 0; text-align: center;">📊 ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH SVM</h1>
        <p style="margin: 10px 0 0 0; text-align: center; opacity: 0.9; font-size: 1.05rem;">
            Phân tích chi tiết hiệu suất mô hình trên tập test
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ────── TÍNH METRICS ──────
    @st.cache_data(show_spinner="⏳ Đang tính metrics…")
    def compute_metrics_safe():
        """Tính metrics từ dataset"""
        try:
            df_eval = pd.read_csv("data/dataset_1000.csv")
            text_col = "comment" if "comment" in df_eval.columns else df_eval.columns[0]
            label_col = "label" if "label" in df_eval.columns else df_eval.columns[-1]
        except Exception as e:
            return None

        X = df_eval[text_col].astype(str).tolist()
        y = df_eval[label_col].tolist()

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(y_test, y_pred),
            "hamming": hamming_loss(y_test, y_pred),
        }

        cm = confusion_matrix(y_test, y_pred)
        
        unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
        label_names_map = {0: "Bình thường", 1: "Tiêu cực"}
        target_names = [label_names_map.get(i, f"Class {i}") for i in unique_labels]
        
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            labels=unique_labels,
            output_dict=True,
            zero_division=0
        )

        return {
            "metrics": metrics,
            "cm": cm,
            "report": report,
            "y_test": y_test,
            "y_pred": y_pred,
            "labels": unique_labels,
            "target_names": target_names,
            "X_test": X_test,
            "X_test_vec": X_test_vec,
        }

    result = compute_metrics_safe()

    # ────── TABS ──────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Metrics", 
        "🔍 Đặc trưng & Tiền xử lý", 
        "📊 Ma trận & Báo cáo",
        "💡 Giải thích ý nghĩa",
        "🎯 Phân tích & Nhận xét"
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 1: METRICS CHÍNH
    # ════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">📈 Các chỉ số đánh giá</div>', unsafe_allow_html=True)
        
        if result:
            col1, col2, col3, col4 = st.columns(4, gap="small")
            
            with col1:
                st.metric("🎯 Accuracy", f"{result['metrics']['accuracy']:.2%}")
                st.caption("Tỷ lệ dự đoán đúng")
            with col2:
                st.metric("⭐ F1-Score", f"{result['metrics']['f1']:.2%}")
                st.caption("Cân bằng Precision & Recall")
            with col3:
                st.metric("🔐 Kappa", f"{result['metrics']['kappa']:.4f}")
                st.caption("Độ đồng ý giữa bộ phân loại")
            with col4:
                st.metric("📉 Hamming", f"{result['metrics']['hamming']:.4f}")
                st.caption("Tỷ lệ lỗi (càng thấp càng tốt)")

            st.divider()

            col_p, col_r = st.columns(2)
            with col_p:
                st.metric("📌 Precision", f"{result['metrics']['precision']:.4f}")
                st.caption("Độ chính xác dương tính")
            with col_r:
                st.metric("🎪 Recall", f"{result['metrics']['recall']:.4f}")
                st.caption("Độ phủ sóng (sensitivity)")

        else:
            st.warning("⚠️ Không thể tính metrics")
            st.info("💡 Hãy chạy trước: python train.py")

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 2: ĐẶC TRƯNG & TIỀN XỬ LÝ
    # ════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">🔍 Đặc trưng sử dụng</div>', unsafe_allow_html=True)

        col_feat_1, col_feat_2 = st.columns(2)
        
        with col_feat_1:
            st.success("**Phương pháp vector hóa:**")
            st.markdown("""
            - **TF-IDF Vectorizer** (Term Frequency - Inverse Document Frequency)
            - Chuyển văn bản thành vector số học
            - Đánh trọng số cao cho các từ có ý nghĩa riêng biệt
            - Loại bỏ ảnh hưởng của các từ phổ biến
            """)
        
        with col_feat_2:
            st.info("**Cấu hình TF-IDF:**")
            st.markdown("""
            - **max_features:** 5000 (tối đa 5000 features)
            - **ngram_range:** (1, 2) (từ đơn + từ kép)
            - **sublinear_tf:** True (logarit hóa tần suất)
            - **min_df:** 2 (từ xuất hiện tối thiểu 2 lần)
            - **max_df:** 0.8 (từ xuất hiện tối đa 80%)
            """)

        st.divider()

        st.markdown('<div class="section-title">🧹 Tiền xử lý dữ liệu</div>', unsafe_allow_html=True)

        steps_prep = [
            ("1️⃣ Chuyển thành chữ thường", "Toàn bộ văn bản → lowercase"),
            ("2️⃣ Loại bỏ URL/Email", "Xóa các đường dẫn web và địa chỉ email"),
            ("3️⃣ Xóa số điện thoại", "Loại bỏ số điện thoại (10+ chữ số)"),
            ("4️⃣ Xóa ký tự đặc biệt", "Giữ lại: a-z, dấu Việt (à, á, â...)"),
            ("5️⃣ Chuẩn hóa khoảng trắng", "Xóa khoảng trắng thừa, trim"),
        ]

        for title, desc in steps_prep:
            st.markdown(f"""
            <div style="background: #f0f9ff; border-radius: 8px; padding: 12px 16px; border-left: 4px solid #3b82f6; margin-bottom: 8px;">
                <strong>{title}</strong><br>
                <span style="color: #475569; font-size: 0.9rem;">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="section-title">⚖️ Chuẩn hóa dữ liệu</div>', unsafe_allow_html=True)
        
        col_norm1, col_norm2 = st.columns(2)
        
        with col_norm1:
            st.warning("**Xử lý Class Imbalance:**")
            st.markdown("""
            - **Bình thường:** ~67% (717 mẫu)
            - **Tiêu cực:** ~33% (358 mẫu)
            - **Tỉ lệ:** 2:1 (không cân bằng)
            - **Giải pháp:** `class_weight='balanced'` trong SVC
            """)
        
        with col_norm2:
            st.info("**Chia tập dữ liệu:**")
            st.markdown("""
            - **Train:** 80% (860 mẫu)
            - **Test:** 20% (215 mẫu)
            - **Stratified Split:** Giữ tỉ lệ class
            - **Random State:** 42 (tái tạo được kết quả)
            """)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 3: MA TRẬN VÀ BÁO CÁO
    # ════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">🔢 Confusion Matrix</div>', unsafe_allow_html=True)

        if result:
            cm = result["cm"]
            target_names = result["target_names"]

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax,
                cbar_kws={'label': 'Số lượng mẫu'},
                annot_kws={'size': 12, 'weight': 'bold'}
            )
            ax.set_xlabel("Dự đoán (Predicted)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Thực tế (Actual)", fontsize=11, fontweight='bold')
            ax.set_title("Confusion Matrix - Tập Test", fontsize=12, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.markdown("**📖 Giải thích Confusion Matrix:**")
            col_cm1, col_cm2 = st.columns(2)
            
            with col_cm1:
                st.info(f"""
                **✅ Dự đoán đúng:**
                
                **TN (True Negative) = {cm[0,0]}**
                - Dự đoán: Bình thường ✓
                - Thực tế: Bình thường ✓
                
                **TP (True Positive) = {cm[1,1]}**
                - Dự đoán: Tiêu cực ✓
                - Thực tế: Tiêu cực ✓
                """)
            
            with col_cm2:
                st.warning(f"""
                **❌ Dự đoán sai:**
                
                **FP (False Positive) = {cm[0,1]}**
                - Dự đoán: Tiêu cực ✗
                - Thực tế: Bình thường (Báo động sai)
                
                **FN (False Negative) = {cm[1,0]}**
                - Dự đoán: Bình thường ✗
                - Thực tế: Tiêu cực (Bỏ sót)
                """)

        st.divider()

        st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)

        if result:
            report_df = pd.DataFrame(result['report']).transpose().round(4)
            st.dataframe(
                report_df.style.background_gradient(
                    cmap='RdYlGn',
                    subset=['precision', 'recall', 'f1-score'],
                    vmin=0,
                    vmax=1
                ),
                use_container_width=True
            )

            st.caption("""
            - **Precision:** Độ chính xác của dự đoán tích cực (TP / (TP+FP))
            - **Recall:** Tỷ lệ phát hiện các mẫu tích cực thực sự (TP / (TP+FN))
            - **F1-score:** Trung bình hài hòa của Precision & Recall
            - **Support:** Số lượng mẫu thực sự của mỗi class
            """)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 4: GIẢI THÍCH Ý NGHĨA
    # ════════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">💡 Giải thích các chỉ số</div>', unsafe_allow_html=True)

        with st.expander("🎯 **Accuracy (Độ chính xác)**", expanded=True):
            st.markdown("""
            **Công thức:** (TP + TN) / (TP + TN + FP + FN)
            
            **Ý nghĩa:** Tỷ lệ dự đoán đúng trên tổng số dự đoán
            
            **Ví dụ:** Accuracy = 60% nghĩa là mô hình dự đoán đúng 60% bình luận
            
            **Khi nào dùng:** Khi các class cân bằng nhau
            
            **Hạn chế:** Không tốt với dữ liệu imbalanced (1 class chiếm đa số)
            """)

        with st.expander("📌 **Precision (Độ chính xác dương tính)**"):
            st.markdown("""
            **Công thức:** TP / (TP + FP)
            
            **Ý nghĩa:** Trong những dự đoán là "tiêu cực", bao nhiêu % là chính xác
            
            **Ví dụ:** Precision = 80% 
            → Trong 100 bình luận dự đoán tiêu cực, 80 cái thực sự tiêu cực
            
            **Quan trọng:** Khi muốn giảm False Positives (báo động sai)
            
            **Ứng dụng:** Spam detection, fraud detection
            """)

        with st.expander("🎪 **Recall (Độ nhạy / Sensitivity)**"):
            st.markdown("""
            **Công thức:** TP / (TP + FN)
            
            **Ý nghĩa:** Trong tất cả bình luận tiêu cực thực sự, phát hiện được bao nhiêu %
            
            **Ví dụ:** Recall = 70% 
            → Phát hiện được 70% bình luận tiêu cực, bỏ sót 30%
            
            **Quan trọng:** Khi muốn giảm False Negatives (bỏ sót)
            
            **Ứng dụng:** Medical diagnosis, content moderation
            """)

        with st.expander("⭐ **F1-Score (Điểm cân bằng)**"):
            st.markdown("""
            **Công thức:** 2 × (Precision × Recall) / (Precision + Recall)
            
            **Ý nghĩa:** Trung bình hài hòa của Precision và Recall
            
            **Điểm:** Từ 0.0 (tồi tệ nhất) → 1.0 (tốt nhất)
            
            **Khi nào dùng:** Khi cần cân bằng giữa Precision và Recall
            
            **Lợi ích:** Thích hợp với dữ liệu imbalanced
            """)

        with st.expander("🔐 **Cohen's Kappa**"):
            st.markdown("""
            **Công thức:** (p_o - p_e) / (1 - p_e)
            
            **Ý nghĩa:** Đo độ đồng ý giữa bộ phân loại và thực tế (loại bỏ yếu tố ngẫu nhiên)
            
            **Giải thích:**
            - Kappa < 0: Hiệu suất xấu hơn ngẫu nhiên
            - 0.0 - 0.2: Sự đồng ý yếu
            - 0.2 - 0.4: Sự đồng ý công bằng
            - 0.4 - 0.6: Sự đồng ý trung bình
            - 0.6 - 0.8: Sự đồng ý khá tốt
            - 0.8 - 1.0: Sự đồng ý rất tốt
            
            **Ưu điểm:** Tốt hơn Accuracy khi dữ liệu imbalanced
            """)

        with st.expander("📉 **Hamming Loss**"):
            st.markdown("""
            **Công thức:** (FP + FN) / (TP + TN + FP + FN)
            
            **Ý nghĩa:** Tỷ lệ dự đoán sai (lỗi)
            
            **Ví dụ:** Hamming Loss = 0.40 → 40% dự đoán sai
            
            **Lưu ý:** Hamming Loss = 1 - Accuracy
            
            **Cách hiểu:** Càng thấp càng tốt
            """)

    # ════════════════════════════════════════════════════════════════════════════
    # TAB 5: PHÂN TÍCH & NHẬN XÉT (CẬP NHẬT)
    # ════════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">🎯 Phân tích & Nhận xét</div>', unsafe_allow_html=True)

        if result:
            metrics = result['metrics']
            cm = result['cm']
            
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            st.markdown("**📊 PHÂN TÍCH CHI TIẾT:**")
            
            col_ana1, col_ana2 = st.columns(2)
            
            with col_ana1:
                st.success("✅ **Điểm Mạnh:**")
                st.markdown(f"""
                1. **Specificity = {specificity:.2%}** ⭐⭐
                   - Mô hình rất tốt trong việc nhận diện bình luận bình thường
                   - Ít báo động sai (FP = {fp})
                   - Phù hợp cho ứng dụng cần độ chính xác cao
                
                2. **Precision = {metrics['precision']:.4f}** ⭐
                   - Khi dự đoán tiêu cực, độ chính xác cao
                   - ~61% dự đoán tiêu cực là đúng
                
                3. **Kappa = {metrics['kappa']:.4f}** ⭐
                   - Độ đồng ý tốt giữa dự đoán và thực tế
                   - Loại bỏ yếu tố ngẫu nhiên
                
                4. **TF-IDF Features = 1,146** ⭐⭐
                   - Đặc trưng đủ để phân biệt 2 lớp
                   - Không quá phức tạp nhưng đủ chi tiết
                
                5. **Model dễ triển khai**
                   - SVM RBF nhanh và hiệu quả
                   - Pickle files nhẹ, dễ deploy
                """)
            
            with col_ana2:
                st.error("❌ **Điểm Yếu:**")
                st.markdown(f"""
                1. **Recall = {sensitivity:.2%}** ❌❌
                   - Bỏ sót {100-sensitivity*100:.1f}% bình luận tiêu cực
                   - FN = {fn} mẫu không được phát hiện
                   - VẤN ĐỀ LỚNHẤT của mô hình hiện tại
                
                2. **Class Imbalance (2:1)**
                   - Dữ liệu không cân bằng (67% vs 33%)
                   - Mô hình bị thiên vị về class đa số
                   - Cần giải pháp cân bằng
                
                3. **Dữ liệu hạn chế**
                   - Chỉ ~1000 mẫu (nhỏ)
                   - Cần thêm 5000-10000 mẫu
                   - Đặc biệt là bình luận tiêu cực
                
                4. **Overfitting khả năng cao**
                   - Cần kiểm tra training accuracy
                   - Nếu training > 80%, test < 61% → overfitting
                
                5. **F1-Score chỉ 0.6118**
                   - Không đủ tốt cho production
                   - Cần ≥ 0.75 cho ứng dụng thực tế
                """)
            
            st.divider()
            
        # TAB 5: PHÂN TÍCH & NHẬN XÉT (FIX - KHÔNG HTML)
            with tab5:
                st.markdown('<div class="section-title">🎯 Phân tích & Nhận xét</div>', unsafe_allow_html=True)

                if result:
                    metrics = result['metrics']
                    cm = result['cm']
                    
                    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    # ────── Điểm mạnh - Điểm yếu
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.success("✅ **Điểm Mạnh:**")
                        st.markdown(f"""
        - **Specificity = {specificity:.2%}** - Ít báo động sai
        - **Precision = {metrics['precision']:.4f}** - Chính xác khi dự đoán
        - **Kappa = {metrics['kappa']:.4f}** - Độ đồng ý tốt
        - **1,146 Features** - Đủ phân biệt 2 lớp
                        """)
                    
                    with col2:
                        st.error("❌ **Điểm Yếu:**")
                        st.markdown(f"""
        - **Recall = {sensitivity:.2%}** - Bỏ sót {100-sensitivity*100:.1f}%
        - **Class Imbalance** - 67% vs 33%
        - **Dữ liệu hạn chế** - Chỉ ~1000 mẫu
        - **F1 = {metrics['f1']:.4f}** - Chưa đủ tốt
                        """)
                    
                    st.divider()
                    
                    # ────── Bảng kết quả
                    st.subheader("📊 Kết quả huấn luyện:")
                    
                    results_data = {
                        "Chỉ số": ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"],
                        "Giá trị": [
                            f"{metrics['accuracy']:.2%}",
                            f"{metrics['precision']:.4f}",
                            f"{sensitivity:.2%}",
                            f"{specificity:.2%}",
                            f"{metrics['f1']:.4f}"
                        ],
                        "Ý nghĩa": [
                            f"Dự đoán đúng {int(metrics['accuracy']*215)}/215 mẫu",
                            "Độ chính xác khi dự đoán tiêu cực",
                            f"⚠️ Bỏ sót {100-sensitivity*100:.1f}%",
                            "✓ Rất tốt",
                            "Cân bằng Precision & Recall"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # ────── Kết luận
                    st.subheader("🎯 Kết luận rõ ràng:")
                    
                    st.success("""
        **✅ Mô hình PHÙ HỢP cho:**
        - Ứng dụng cần Specificity cao (ít báo động sai)
        - Kiểm duyệt bình luận bình thường
        - Lọc nhanh nội dung rõ ràng tiêu cực
                    """)
                    
                    st.error(f"""
        **❌ Mô hình KHÔNG PHÙ HỢP cho:**
        - Phát hiện 100% bình luận tiêu cực (Recall = {sensitivity:.2%} → bỏ sót nhiều)
        - Production-level moderation (cần F1 ≥ 0.75)
        - Ứng dụng cần cân bằng Precision-Recall
                    """)
                    
                    st.divider()
                    
                    # ────── Đề xuất cải thiện
                    st.subheader("📈 Đề xuất cải thiện (Ưu tiên):")
                    
                    suggestions = [
                        ("🔴 1. Thu thập thêm dữ liệu tiêu cực", 
                        "Tăng từ ~350 lên 1000-2000 mẫu tiêu cực → Cân bằng dataset & tăng Recall"),
                        
                        ("🟠 2. Sử dụng SMOTE oversampling", 
                        "Cân bằng dữ liệu training → Tăng Recall mà không cần dữ liệu mới"),
                        
                        ("🟡 3. Điều chỉnh class_weight/threshold", 
                        "Tăng cost cho False Negatives → Hoặc giảm threshold để tăng Recall"),
                        
                        ("🟢 4. Thử các mô hình khác", 
                        "XGBoost, LightGBM, LSTM, BERT → Ensemble methods"),
                        
                        ("🟢 5. Feature engineering", 
                        "Sentiment lexicon, Word2Vec, FastText → N-gram tiếng Việt tối ưu"),
                        
                        ("🔵 6. Ensemble methods", 
                        "SVM + Random Forest + Naive Bayes → Voting classifier")
                    ]
                    
                    for title, desc in suggestions:
                        st.info(f"**{title}**\n\n{desc}")
                    
                    st.divider()
        else:
            st.warning("⚠️ Không thể tính metrics")
# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Đồ án: Phân loại bình luận tiếng Việt bằng SVM • Nguyễn Thành Nhật • MSSV: 21T1020563")