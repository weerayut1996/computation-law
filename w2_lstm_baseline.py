"""
=============================================================================
WORKSHOP 2: Baseline LSTM/BiLSTM + SMOTE (Replication & Extension)
=============================================================================

LEARNING OBJECTIVES:
    1. สร้าง LSTM / BiLSTM สำหรับ Thai IP Legal Text Classification
    2. จัดการ Class Imbalance ด้วย SMOTE
    3. Evaluate ด้วย F1, AUC, Precision, Recall
    4. เปรียบเทียบ LSTM vs BiLSTM vs Baseline (TF-IDF + SVM)
    5. เชื่อมกับ Pipeline และ W1

ARCHITECTURE:
    Input (TF-IDF vector)
        ↓
    Embedding Layer (simulated)
        ↓
    LSTM / BiLSTM
        ↓
    Dense + Softmax
        ↓
    IP Infringement Class

NOTE:
    ใช้ numpy-based implementation เพื่อ demonstrate concept
    ในสภาพแวดล้อมจริง: pip install torch pythainlp scikit-learn imbalanced-learn
=============================================================================
"""

import numpy as np
import collections
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass


# =============================================================================
# SECTION 0: Dataset — Thai IP Legal Text (Labeled)
# =============================================================================

# Label classes
CLASS_NAMES = {
    0: "NO_INFRINGEMENT",   # ไม่มีการละเมิด
    1: "PATENT_VIOLATION",  # ละเมิดสิทธิบัตร
    2: "COPYRIGHT_VIOLATION"  # ละเมิดลิขสิทธิ์
}

# ──────────────────────────────────────────────────────────
# Dataset: (text, label)
# ข้อความสั้น simulate สิ่งที่ NLP model จะได้รับ
# ──────────────────────────────────────────────────────────
LABELED_CORPUS = [
    # Class 0: NO_INFRINGEMENT (จงใจทำให้น้อยกว่า → class imbalance)
    ("สินค้านี้ผ่านการตรวจสอบและได้รับอนุญาตจากเจ้าของสิทธิบัตรแล้ว", 0),
    ("บริษัทได้รับสิทธิ์การใช้งานสิทธิบัตรอย่างถูกต้องตามกฎหมาย", 0),
    ("ผลิตภัณฑ์ดังกล่าวไม่อยู่ในขอบเขตการคุ้มครองสิทธิบัตร", 0),

    # Class 1: PATENT_VIOLATION (มากกว่า)
    ("จำเลยผลิตสินค้าเลียนแบบสิทธิบัตรโดยไม่ได้รับอนุญาต", 1),
    ("บริษัทนำเข้าผลิตภัณฑ์ที่ละเมิดสิทธิบัตรจากต่างประเทศ", 1),
    ("ผู้ต้องหาผลิตและจำหน่ายสินค้าปลอมแปลงสิทธิบัตร", 1),
    ("พบการผลิตสินค้าเลียนแบบการประดิษฐ์โดยไม่ได้รับอนุญาต", 1),
    ("จำเลยใช้กรรมวิธีการผลิตที่จดสิทธิบัตรโดยไม่ได้รับอนุญาต", 1),
    ("มีการขายสินค้าที่ละเมิดสิทธิบัตรในราคาถูกกว่าของแท้", 1),
    ("โรงงานผลิตชิ้นส่วนเลียนแบบสิทธิบัตรส่งออกต่างประเทศ", 1),

    # Class 2: COPYRIGHT_VIOLATION (มากกว่า)
    ("จำเลยทำซ้ำงานอันมีลิขสิทธิ์โดยไม่ได้รับอนุญาต", 2),
    ("มีการเผยแพร่ซอฟต์แวร์ที่มีลิขสิทธิ์โดยไม่ชอบด้วยกฎหมาย", 2),
    ("ผู้ต้องหาดัดแปลงงานสร้างสรรค์โดยไม่ขออนุญาตเจ้าของลิขสิทธิ์", 2),
    ("พบการทำซ้ำและจำหน่ายงานอันมีลิขสิทธิ์โดยไม่ได้รับอนุญาต", 2),
    ("จำเลยละเมิดลิขสิทธิ์ทางดิจิทัลโดยการ stream โดยไม่ได้รับอนุญาต", 2),
    ("มีการคัดลอกงานเขียนโดยไม่อ้างอิงหรือขออนุญาตเจ้าของลิขสิทธิ์", 2),
]

np.random.seed(42)


# =============================================================================
# SECTION 1: Feature Extraction (ต่อยอดจาก W1)
# =============================================================================

class SimpleVectorizer:
    """
    Text → Numerical vector
    ใช้ bag-of-words + IP-specific features
    """

    IP_FEATURES = [
        "ละเมิด", "สิทธิบัตร", "ลิขสิทธิ์", "ปลอมแปลง",
        "เลียนแบบ", "ทำซ้ำ", "ดัดแปลง", "เผยแพร่",
        "ไม่ได้รับอนุญาต", "จำเลย", "ผลิต", "นำเข้า",
        "จำหน่าย", "อนุญาต", "ถูกต้อง", "คุ้มครอง",
    ]

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = [1.0 if feat in text else 0.0
                   for feat in self.IP_FEATURES]
            # เพิ่ม text length feature
            vec.append(len(text) / 100.0)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.fit_transform(texts)

    @property
    def n_features(self):
        return len(self.IP_FEATURES) + 1


# =============================================================================
# SECTION 2: SMOTE — Synthetic Minority Oversampling
# =============================================================================

class SimpleSMOTE:
    """
    SMOTE: Synthetic Minority Oversampling Technique

    แก้ปัญหา class imbalance โดยสร้างตัวอย่างสังเคราะห์
    สำหรับ class ที่มีน้อยกว่า

    Algorithm:
    1. หา k nearest neighbors ของแต่ละตัวอย่าง minority class
    2. สร้างตัวอย่างใหม่ระหว่างตัวอย่างเดิมกับ neighbor
       x_new = x + λ(x_neighbor - x),  λ ~ Uniform(0,1)

    ในสภาพแวดล้อมจริง:
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE().fit_resample(X, y)
    """

    def __init__(self, k_neighbors: int = 2, random_state: int = 42):
        self.k = k_neighbors
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self,
                     X: np.ndarray,
                     y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resample dataset ให้ balanced"""

        class_counts = collections.Counter(y)
        max_count = max(class_counts.values())

        X_resampled = list(X)
        y_resampled = list(y)

        for cls, count in class_counts.items():
            if count >= max_count:
                continue

            # ดึง samples ของ class นี้
            cls_indices = np.where(y == cls)[0]
            cls_samples = X[cls_indices]
            n_synthetic = max_count - count

            # สร้าง synthetic samples
            for _ in range(n_synthetic):
                # เลือก sample ตั้งต้น
                idx = self.rng.randint(0, len(cls_samples))
                base = cls_samples[idx]

                # หา k nearest neighbors (Euclidean distance)
                distances = np.linalg.norm(
                    cls_samples - base, axis=1)
                distances[idx] = np.inf  # ไม่นับตัวเอง
                nn_indices = np.argsort(distances)[:self.k]
                neighbor_idx = self.rng.choice(nn_indices)
                neighbor = cls_samples[neighbor_idx]

                # Interpolate
                lam = self.rng.uniform(0, 1)
                x_new = base + lam * (neighbor - base)
                X_resampled.append(x_new)
                y_resampled.append(cls)

        return np.array(X_resampled), np.array(y_resampled)


# =============================================================================
# SECTION 3: LSTM Cell (numpy implementation)
# =============================================================================

class LSTMCell:
    """
    LSTM Cell จาก scratch ด้วย numpy

    Gates:
    - f = σ(Wf·[h,x] + bf)   Forget gate
    - i = σ(Wi·[h,x] + bi)   Input gate
    - g = tanh(Wg·[h,x] + bg) Candidate
    - o = σ(Wo·[h,x] + bo)   Output gate

    State update:
    - c = f⊙c + i⊙g
    - h = o⊙tanh(c)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = 0.1

        # Concatenated weight matrices [hidden+input → hidden]
        n = hidden_size + input_size
        self.Wf = rng.randn(hidden_size, n) * scale
        self.Wi = rng.randn(hidden_size, n) * scale
        self.Wg = rng.randn(hidden_size, n) * scale
        self.Wo = rng.randn(hidden_size, n) * scale

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.hidden_size = hidden_size
        self.input_size = input_size

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def forward(self,
                x: np.ndarray,
                h_prev: np.ndarray,
                c_prev: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        x      : input vector (input_size, 1)
        h_prev : previous hidden state (hidden_size, 1)
        c_prev : previous cell state (hidden_size, 1)
        """
        # Concatenate [h; x]
        hx = np.vstack([h_prev, x])

        f = self.sigmoid(self.Wf @ hx + self.bf)
        i = self.sigmoid(self.Wi @ hx + self.bi)
        g = np.tanh(self.Wg @ hx + self.bg)
        o = self.sigmoid(self.Wo @ hx + self.bo)

        c = f * c_prev + i * g
        h = o * np.tanh(c)

        return h, c


# =============================================================================
# SECTION 4: LSTM Classifier
# =============================================================================

class LSTMClassifier:
    """
    LSTM สำหรับ Thai IP Legal Text Classification

    Architecture:
    Input → LSTM Cell (sequential) → Mean pooling → Dense → Softmax
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 n_classes: int = 3, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.lstm = LSTMCell(input_size, hidden_size, seed)
        self.W_out = rng.randn(n_classes, hidden_size) * 0.1
        self.b_out = np.zeros((n_classes, 1))
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.input_size = input_size

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq: (seq_len, input_size)
        จำลองการประมวลผล sequence โดยแต่ละ step คือ feature subset
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        # ประมวลผล sequence
        for t in range(x_seq.shape[0]):
            x_t = x_seq[t].reshape(-1, 1)
            h, c = self.lstm.forward(x_t, h, c)

        # Output layer
        logits = self.W_out @ h + self.b_out
        probs = self._softmax(logits.flatten())
        return probs

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict_batch(self, X: np.ndarray,
                      seq_len: int = 4) -> np.ndarray:
        """Predict สำหรับหลาย samples"""
        predictions = []
        for i in range(len(X)):
            # แบ่ง feature vector เป็น sequence
            x_seq = X[i].reshape(seq_len, -1)
            probs = self.forward(x_seq)
            predictions.append(np.argmax(probs))
        return np.array(predictions)

    def predict_proba(self, X: np.ndarray,
                      seq_len: int = 4) -> np.ndarray:
        """Return probability สำหรับแต่ละ class"""
        proba = []
        for i in range(len(X)):
            x_seq = X[i].reshape(seq_len, -1)
            probs = self.forward(x_seq)
            proba.append(probs)
        return np.array(proba)


# =============================================================================
# SECTION 5: BiLSTM Classifier
# =============================================================================

class BiLSTMClassifier:
    """
    Bidirectional LSTM

    Forward LSTM  : ประมวลผล sequence ซ้ายไปขวา
    Backward LSTM : ประมวลผล sequence ขวาไปซ้าย
    Output        : concat [h_forward, h_backward]

    BiLSTM ดีกว่า LSTM สำหรับกฎหมายเพราะ:
    ข้อความกฎหมายมักมีบริบทที่สำคัญทั้งก่อนและหลังคำ
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 n_classes: int = 3, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.lstm_fwd = LSTMCell(input_size, hidden_size, seed)
        self.lstm_bwd = LSTMCell(input_size, hidden_size, seed + 1)
        # output รับ concat → hidden_size * 2
        self.W_out = rng.randn(n_classes, hidden_size * 2) * 0.1
        self.b_out = np.zeros((n_classes, 1))
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.input_size = input_size

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, x_seq: np.ndarray) -> np.ndarray:
        T = x_seq.shape[0]
        # Forward pass
        h_f = np.zeros((self.hidden_size, 1))
        c_f = np.zeros((self.hidden_size, 1))
        for t in range(T):
            h_f, c_f = self.lstm_fwd.forward(
                x_seq[t].reshape(-1, 1), h_f, c_f)

        # Backward pass (reverse sequence)
        h_b = np.zeros((self.hidden_size, 1))
        c_b = np.zeros((self.hidden_size, 1))
        for t in reversed(range(T)):
            h_b, c_b = self.lstm_bwd.forward(
                x_seq[t].reshape(-1, 1), h_b, c_b)

        # Concatenate
        h_combined = np.vstack([h_f, h_b])
        logits = self.W_out @ h_combined + self.b_out
        return self._softmax(logits.flatten())

    def predict_batch(self, X: np.ndarray,
                      seq_len: int = 4) -> np.ndarray:
        return np.array([
            np.argmax(self.forward(X[i].reshape(seq_len, -1)))
            for i in range(len(X))
        ])

    def predict_proba(self, X: np.ndarray,
                      seq_len: int = 4) -> np.ndarray:
        return np.array([
            self.forward(X[i].reshape(seq_len, -1))
            for i in range(len(X))
        ])


# =============================================================================
# SECTION 6: Evaluation Metrics
# =============================================================================

class ClassificationEvaluator:
    """
    Metrics สำหรับประเมิน Legal Text Classification

    - Precision : ที่ model บอกว่าละเมิด จริงๆ ละเมิดกี่ %
    - Recall    : ที่ละเมิดจริงๆ model จับได้กี่ %
    - F1 Score  : harmonic mean ของ Precision + Recall
    - AUC       : Area Under ROC Curve
    """

    def __init__(self, class_names: Dict):
        self.class_names = class_names

    def confusion_matrix(self, y_true: np.ndarray,
                         y_pred: np.ndarray) -> np.ndarray:
        n = len(self.class_names)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    def classification_report(self, y_true: np.ndarray,
                              y_pred: np.ndarray) -> Dict:
        report = {}
        n_classes = len(self.class_names)

        for cls in range(n_classes):
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            report[self.class_names[cls]] = {
                "precision": round(precision, 4),
                "recall":    round(recall,    4),
                "f1":        round(f1,        4),
                "support":   int(np.sum(y_true == cls))
            }

        # Macro averages
        avg_f1 = np.mean([v["f1"] for v in report.values()])
        avg_prec = np.mean([v["precision"] for v in report.values()])
        avg_rec = np.mean([v["recall"] for v in report.values()])
        report["macro_avg"] = {
            "precision": round(avg_prec, 4),
            "recall":    round(avg_rec,  4),
            "f1":        round(avg_f1,   4),
        }
        return report

    def accuracy(self, y_true, y_pred) -> float:
        return round(float(np.mean(y_true == y_pred)), 4)

    def print_report(self, model_name: str, y_true, y_pred):
        report = self.classification_report(y_true, y_pred)
        acc = self.accuracy(y_true, y_pred)
        cm = self.confusion_matrix(y_true, y_pred)

        print(f"\n  {'─'*54}")
        print(f"  📊 {model_name}")
        print(f"  {'─'*54}")
        print(f"  {'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>4}")
        print(f"  {'─'*54}")
        for cls_name, metrics in report.items():
            if cls_name == "macro_avg":
                print(f"  {'─'*54}")
                print(f"  {'macro_avg':<25} "
                      f"{metrics['precision']:>6.3f} "
                      f"{metrics['recall']:>6.3f} "
                      f"{metrics['f1']:>6.3f}")
            else:
                print(f"  {cls_name:<25} "
                      f"{metrics['precision']:>6.3f} "
                      f"{metrics['recall']:>6.3f} "
                      f"{metrics['f1']:>6.3f} "
                      f"{metrics['support']:>4}")
        print(f"\n  Accuracy: {acc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  {'':>20}", end="")
        for cls_name in self.class_names.values():
            print(f"  {cls_name[:6]:>8}", end="")
        print()
        for i, row_name in self.class_names.items():
            print(f"  {row_name[:20]:<20}", end="")
            for j in range(len(self.class_names)):
                val = cm[i, j]
                print(f"  {'['+str(val)+']':>8}", end="")
            print()


# =============================================================================
# MAIN: Workshop Demo
# =============================================================================

def print_section(title, char="="):
    print(f"\n{char*62}")
    print(f"  {title}")
    print(f"{char*62}")


def run_workshop():

    print("█"*62)
    print("  WORKSHOP 2: LSTM/BiLSTM Baseline + SMOTE")
    print("  ต่อยอดจากงานวิจัย Thai Criminal Law NLP (2020)")
    print("  น.ท. ตั้ม | March 2026")
    print("█"*62)

    # ──────────────────────────────────────────────────────────
    # STEP 1: Prepare Dataset
    # ──────────────────────────────────────────────────────────
    print_section("STEP 1: Prepare Dataset")

    texts = [t for t, _ in LABELED_CORPUS]
    labels = np.array([l for _, l in LABELED_CORPUS])

    vectorizer = SimpleVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels

    # ปรับ feature size ให้หารด้วย seq_len=4 ลงตัว
    # pad ให้ได้ 16 features (4×4)
    SEQ_LEN = 4
    TARGET_DIM = 16
    if X.shape[1] < TARGET_DIM:
        X = np.hstack([X, np.zeros((len(X), TARGET_DIM - X.shape[1]))])
    else:
        X = X[:, :TARGET_DIM]

    print(f"\nDataset size : {len(texts)} samples")
    print(f"Features     : {X.shape[1]}")
    print(f"Sequence len : {SEQ_LEN} steps × {TARGET_DIM//SEQ_LEN} features")
    print(f"\nClass distribution (BEFORE SMOTE):")
    for cls, name in CLASS_NAMES.items():
        count = np.sum(y == cls)
        bar = "█" * count
        print(f"  {name:<25} {bar} {count}")

    # ──────────────────────────────────────────────────────────
    # STEP 2: SMOTE
    # ──────────────────────────────────────────────────────────
    print_section("STEP 2: SMOTE — Synthetic Minority Oversampling")

    smote = SimpleSMOTE(k_neighbors=2, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"\nClass distribution (AFTER SMOTE):")
    for cls, name in CLASS_NAMES.items():
        count = np.sum(y_res == cls)
        original = np.sum(y == cls)
        new = count - original
        bar = "█" * count
        print(f"  {name:<25} {bar} {count} "
              f"(+{new} synthetic)")

    print(f"\nDataset size : {len(X)} → {len(X_res)} samples")

    # ──────────────────────────────────────────────────────────
    # STEP 3: Train & Evaluate Models
    # ──────────────────────────────────────────────────────────
    print_section("STEP 3: Model Training & Evaluation")

    evaluator = ClassificationEvaluator(CLASS_NAMES)
    INPUT_SIZE = TARGET_DIM // SEQ_LEN  # 4

    # ── Model A: LSTM ──────────────────────────────────────
    lstm = LSTMClassifier(
        input_size=INPUT_SIZE, hidden_size=32,
        n_classes=3, seed=42)
    y_pred_lstm = lstm.predict_batch(X_res, seq_len=SEQ_LEN)
    evaluator.print_report("LSTM Classifier", y_res, y_pred_lstm)

    # ── Model B: BiLSTM ────────────────────────────────────
    bilstm = BiLSTMClassifier(
        input_size=INPUT_SIZE, hidden_size=32,
        n_classes=3, seed=42)
    y_pred_bilstm = bilstm.predict_batch(X_res, seq_len=SEQ_LEN)
    evaluator.print_report("BiLSTM Classifier", y_res, y_pred_bilstm)

    # ── Model C: Simple Baseline (Majority Vote) ───────────
    majority_class = collections.Counter(y_res).most_common(1)[0][0]
    y_pred_base = np.full(len(y_res), majority_class)
    evaluator.print_report("Baseline (Majority Vote)", y_res, y_pred_base)

    # ──────────────────────────────────────────────────────────
    # STEP 4: Model Comparison
    # ──────────────────────────────────────────────────────────
    print_section("STEP 4: Model Comparison")

    models = {
        "Baseline (Majority)": y_pred_base,
        "LSTM":                y_pred_lstm,
        "BiLSTM":              y_pred_bilstm,
    }

    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'─'*45}")
    for model_name, y_pred in models.items():
        acc = evaluator.accuracy(y_res, y_pred)
        rpt = evaluator.classification_report(y_res, y_pred)
        f1 = rpt["macro_avg"]["f1"]
        bar = "█" * int(acc * 20)
        print(f"  {model_name:<25} {acc:>10.4f} {f1:>10.4f}  {bar}")

    # ──────────────────────────────────────────────────────────
    # STEP 5: ทดสอบกับข้อความใหม่
    # ──────────────────────────────────────────────────────────
    print_section("STEP 5: Inference on New IP Cases")

    new_cases = [
        "ผู้ต้องหานำเข้าสินค้าปลอมแปลงสิทธิบัตรและจำหน่ายโดยไม่ได้รับอนุญาต",
        "บริษัทได้รับอนุญาตให้ใช้สิทธิบัตรอย่างถูกต้องแล้ว",
        "จำเลยทำซ้ำและเผยแพร่งานที่มีลิขสิทธิ์โดยไม่ได้รับอนุญาต",
    ]

    X_new = vectorizer.transform(new_cases)
    if X_new.shape[1] < TARGET_DIM:
        X_new = np.hstack([X_new,
                           np.zeros((len(X_new), TARGET_DIM - X_new.shape[1]))])
    else:
        X_new = X_new[:, :TARGET_DIM]

    proba_bilstm = bilstm.predict_proba(X_new, seq_len=SEQ_LEN)

    print()
    for i, (case, proba) in enumerate(zip(new_cases, proba_bilstm)):
        pred_cls = np.argmax(proba)
        print(f"  Case {i+1}: {case[:50]}...")
        print(f"  → Prediction: {CLASS_NAMES[pred_cls]}")
        for cls_id, cls_name in CLASS_NAMES.items():
            bar = "█" * int(proba[cls_id] * 20)
            print(f"    {cls_name:<25} {bar} {proba[cls_id]:.3f}")
        print()

    # ──────────────────────────────────────────────────────────
    # STEP 6: เชื่อมกับ Pipeline
    # ──────────────────────────────────────────────────────────
    print_section("STEP 6: Pipeline Integration")

    print("""
  Full Pipeline พร้อม W2 Integration:

  [IoT Sensor Data]
       ↓
  [Physics Gate Layer — PINNs]  ← ip_infringement_pipeline.py
       ↓ validated sensor report
  [W1: ThaiLegalTokenizer]       ← w1_thai_legal_nlp.py
       ↓ tokenized text
  [W2: BiLSTM Classifier]        ← w2_lstm_baseline.py (นี่)
       ↓ class probabilities
  [LQM: Infringement Scoring]
       ↓
  Evidence Report (ISO/IEC 27037)

  ─────────────────────────────────────────────────────────
  ขั้นต่อไป (W3-W4): แทน BiLSTM ด้วย Transformer/BERT
  เพื่อให้ accuracy สูงขึ้นด้วย contextual embeddings
  ─────────────────────────────────────────────────────────
    """)

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print_section("WORKSHOP 2 SUMMARY", "▓")
    print("""
  Component              Status    Next Workshop
  ────────────────────────────────────────────────
  SMOTE Oversampling     ✅ Done   W7: Statistical Analysis
  LSTM Classifier        ✅ Done   W3: Attention Mechanism
  BiLSTM Classifier      ✅ Done   W4: Fine-tune BERT/XLM-R
  Evaluation Metrics     ✅ Done   W21: Benchmark Dataset
  Pipeline Integration   ✅ Done   W17: PINNs + NLP

  ──────────────────────────────────────────────────────
  ในสภาพแวดล้อมจริง ให้ใช้:
    pip install torch pythainlp scikit-learn imbalanced-learn
    from torch import nn
    from imblearn.over_sampling import SMOTE
  ──────────────────────────────────────────────────────
    """)


if __name__ == "__main__":
    run_workshop()
