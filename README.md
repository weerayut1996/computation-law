# Computation Law Research Project ⚖️🤖
งานวิจัยการประยุกต์ใช้ AI ในทางกฎหมาย โดยเน้นการบูรณาการระหว่าง **NLP**, **PINNs**, และ **LQM**

## 🔬 Research Focus
- **Natural Language Processing (NLP):** วิเคราะห์ข้อความกฎหมายไทย
- **Physics-Informed Neural Networks (PINNs):** ประมวลผลข้อเท็จจริงจากเหตุการณ์จริง (Camera-based)
- **Large Quantitative Models (LQM):** คำนวณอัตราโทษและพฤติการณ์เชิงปริมาณ

## 📂 Project Structure
# ⚖️ Computation Law Research Project
**การประยุกต์ใช้ AI และโครงข่ายประสาทเทียมในงานกฎหมายไทย (Thai Computational Law)**

โปรเจกต์นี้มุ่งเน้นการพัฒนาระบบประมวลผลภาษาธรรมชาติ (NLP) สำหรับข้อมูลกฎหมายไทย โดยบูรณาการระหว่างกฎเกณฑ์ทางกฎหมายและโมเดลการเรียนรู้เชิงลึก (Deep Learning) เพื่อรองรับการขยายผลสู่ระบบ PINNs และ LQM ในอนาคต

---

## 📂 Project Structure & Methodology

### 🛠️ 1. w1_thai_legal_nlp.py (Foundation Layer)
**หัวข้อหลัก: การสกัดโครงสร้างข้อมูลจากภาษากฎหมายไทย (Legal Information Extraction)**

* **1.1 Domain-Specific Tokenization:** * แก้ปัญหา "คำเฉพาะ" โดยใช้ `LEGAL_COMPOUNDS` เพื่อรักษาความหมายทางกฎหมาย (**Semantic Integrity**) ป้องกันไม่ให้คำสำคัญ (เช่น *สิทธิบัตรการประดิษฐ์*) ถูกตัดแยกจนเสียรูปคดีหรือลดทอนความสำคัญเชิงนิติศาสตร์
* **1.2 NER Lite (Regex-based):** * การแปลง **Unstructured Text** ให้เป็น **Structured Data** โดยเน้นการดึงข้อมูลตัวเลข (Penalty/Fine) และ Entity ทางกฎหมาย เพื่อป้อนเข้าสู่ชั้นการคำนวณเชิงปริมาณ (**LQM - Large Quantitative Models**)
* **1.3 Semantic Co-occurrence:** * การพิสูจน์สมมติฐานว่า "บริบทของคำกฎหมายมีรูปแบบที่แน่นอน" ผ่าน **Co-occurrence Matrix** ซึ่งเป็นรากฐานสำคัญของระบบ Search และ Recommendation ในงานกฎหมาย

---

### 🧠 2. w2_lstm_baseline.py (Deep Learning Layer)
**หัวข้อหลัก: การจำแนกประเภทข้อพิพาทด้วยโครงข่ายประสาทเทียม (Legal Text Classification)**

* **2.1 Data Augmentation (SMOTE):** * การแก้ปัญหาข้อมูลน้อย (**Small Data**) ในคดีเฉพาะทาง โดยใช้การสังเคราะห์ข้อมูลใหม่ด้วยหลักเรขาคณิต เพื่อลดความลำเอียงของ AI (**Algorithmic Bias**) และเพิ่มประสิทธิภาพในการตรวจจับคดีที่เป็นส่วนน้อย (Minority Class)
* **2.2 Sequence Modeling (LSTM/BiLSTM):**
    * **LSTM:** ใช้ระบบ **Gate Control** (Forget/Input/Output) เพื่อจดจำเงื่อนไขทางกฎหมายที่ยาวและซับซ้อน ป้องกันปัญหาการลืมบริบทต้นประโยค (Vanishing Gradient)
    * **BiLSTM:** เพิ่มประสิทธิภาพด้วยการประมวลผลแบบสองทิศทาง (**Forward-Backward**) เพื่อเก็บบริบท "หน้า-หลัง" ของคำขยายในประโยคกฎหมายไทย ทำให้ค่า **F1-Score** สูงกว่าโมเดลทิศทางเดียวอย่างมีนัยสำคัญ
* **2.3 Legal-Centric Evaluation:** * การวัดผลที่เน้น **Precision & Recall** มากกว่า Accuracy เพื่อลดความเสี่ยงในการ "กล่าวหาผิด" (**False Positive**) หรือ "ตรวจไม่เจอ" (**False Negative**) ซึ่งส่งผลกระทบโดยตรงต่อกระบวนการยุติธรรม

---

## 🚀 เทคโนโลยีที่ใช้งาน (Tech Stack)
* **Language:** Python 3.14+
* **Deep Learning:** PyTorch (MPS Accelerated for Mac Mr9T)
* **NLP Tools:** PyThaiNLP, Scikit-learn
* **Augmentation:** Imbalanced-learn (SMOTE)

---
*จัดทำโดย: นาวาอากาศโท ตั้ม (Wing Commander) | PhD Candidate in Computational Law*


## note
1. note-w1.txt
สรุปผลการดำเนินงาน Workshop 1 และการตั้งค่า Environment (venv_comp_law)
2. note-w2.txt
สรุปผลการเปรียบเทียบประสิทธิภาพระหว่าง LSTM vs BiLSTM และตารางค่า Metrics ที่ได้จากการทำ SMOTE เพื่อใช้เป็น Baseline สำหรับ PhD Thesis

## 🛠️ Tech Stack
- **Language:** Python 3.14+
- **Frameworks:** PyTorch (MPS Accelerated), PyThaiNLP
- **VCS:** Git & GitHub