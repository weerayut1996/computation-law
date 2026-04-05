"""
=============================================================================
WORKSHOP 1: Thai Legal Text Processing
=============================================================================
LEARNING OBJECTIVES:
    1. Tokenization ภาษาไทยสำหรับข้อความกฎหมาย IP (Intellectual Property:ทรัพย์สินทางปัญญา)
    2. Word Embeddings (TF-IDF + Co-occurrence Matrix)
    3. Legal Entity Extraction (มาตรา, ประเภทความผิด, โทษ)
    4. Keyword Analysis สำหรับ Patent & Copyright
    5. เชื่อมกับ Pipeline ที่สร้างไว้ใน PoC

PACKAGES:
    - สภาพแวดล้อมนี้ : numpy, scipy, re, collections (built-in)
    - สภาพแวดล้อมจริง: pip install pythainlp transformers torch

DATASET:
    ใช้ข้อความจาก พ.ร.บ. สิทธิบัตร พ.ศ. 2522 และ พ.ร.บ. ลิขสิทธิ์ พ.ศ. 2537 (ตัวอย่างข้อความจริง)
=============================================================================
"""
import os
import re
import json
import numpy as np
import collections
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
# =============================================================================
# SECTION 0: Dataset — โหลดข้อมูลจาก JSON (แทนที่ข้อความเดิม)
# =============================================================================
# 1. กำหนดตำแหน่งไฟล์ให้แม่นยำ (อ้างอิงจากตำแหน่งไฟล์ w1 นี้)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'thai_ip_corpus.json')


def load_legal_corpus(file_path):
    """ฟังก์ชันสำหรับอ่านข้อมูลจาก JSON เข้าสู่โปรแกรม"""
    if not os.path.exists(file_path):
        print(
            f"❌ ไม่พบไฟล์ที่: {file_path} (กรุณารัน convert_to_json.py ก่อนครับ)")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # ดึงเฉพาะฟีลด์ 'text' ของทุกรายการมาสร้างเป็น List
        return [item['text'] for item in data]


# 2. นำข้อมูลมาเก็บในตัวแปรเดิมเพื่อให้ระบบส่วนที่เหลือทำงานต่อได้
THAI_IP_CORPUS = load_legal_corpus(DATA_PATH)

# ตรวจสอบจำนวนข้อมูลที่โหลดได้
print(f"✅ Loaded {len(THAI_IP_CORPUS)} documents from {DATA_PATH}")

# =============================================================================
# SECTION 1: Thai Tokenizer
# =============================================================================


class ThaiLegalTokenizer:
    """
    Tokenizer สำหรับข้อความกฎหมายไทย

    ในสภาพแวดล้อมจริง ให้ใช้:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(text, engine='newmm')

    ที่นี่ใช้ rule-based tokenizer ที่เหมาะกับข้อความกฎหมายครับ
    """

    # คำสำคัญในกฎหมาย IP ที่ต้องเก็บไว้เป็น unit เดียว
    LEGAL_COMPOUNDS = [
        "สิทธิบัตรการประดิษฐ์", "ลิขสิทธิ์", "เครื่องหมายการค้า",
        "ทรัพย์สินทางปัญญา", "การละเมิดสิทธิ", "การประดิษฐ์ขึ้นใหม่",
        "ขั้นการประดิษฐ์", "ทางอุตสาหกรรม", "จำคุก", "ปรับ",
        "ริบทรัพย์", "เจ้าของสิทธิ", "ผู้ทรงสิทธิ", "คำขอรับสิทธิบัตร",
        "การดัดแปลง", "การเผยแพร่", "การทำซ้ำ", "พระราชบัญญัติ",
    ]

    # Stop words ภาษาไทย (กฎหมาย)
    STOP_WORDS = {
        "และ", "หรือ", "ที่", "ใน", "ของ", "ตาม", "โดย", "กับ",
        "แก่", "ซึ่ง", "เพื่อ", "จาก", "ไว้", "ได้", "มี", "เป็น",
        "ต้อง", "ให้", "แห่ง", "นี้", "นั้น", "ว่า", "แต่", "ทั้ง",
        "ดัง", "ดังนี้", "ตั้งแต่", "ถึง", "หรือ", "ทั้งจำทั้งปรับ",
    }

    def __init__(self):
        # เรียงจากยาวไปสั้น เพื่อ match compound words ก่อน
        self.compounds = sorted(self.LEGAL_COMPOUNDS,
                                key=len, reverse=True)

    def tokenize(self, text: str,
                 remove_stopwords: bool = False) -> List[str]:
        """
        Tokenize ข้อความกฎหมายไทย

        Algorithm:
        1. ป้องกัน compound words ก่อน
        2. แยกด้วย pattern-based rules
        3. กรอง stop words (ถ้าต้องการ)
        """
        # Step 1: แทน compound words ด้วย placeholder
        protected = text
        placeholders = {}
        for i, compound in enumerate(self.compounds):
            if compound in protected:
                ph = f"__COMPOUND_{i}__"
                placeholders[ph] = compound
                protected = protected.replace(compound, ph)

        # Step 2: แยกโดย pattern
        # ตัดที่ space, วรรค, และ boundary ของคำไทย
        raw_tokens = re.split(
            r'[\s\u0020\u00a0\u3000]+|'   # whitespace
            r'(?<=[ก-๙])(?=[A-Za-z0-9])|'  # Thai->Latin
            r'(?<=[A-Za-z0-9])(?=[ก-๙])',  # Latin->Thai
            protected
        )

        # Step 3: คืน compound words กลับ
        tokens = []
        for tok in raw_tokens:
            tok = tok.strip()
            if not tok:
                continue
            # คืน placeholder
            resolved = placeholders.get(tok, tok)
            # แยกเครื่องหมายวรรคตอนออก
            parts = re.split(r'([,\.\(\)\[\]๑-๙0-9]+)', resolved)
            for p in parts:
                p = p.strip()
                if p:
                    tokens.append(p)

        # Step 4: กรอง stop words
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.STOP_WORDS]

        return [t for t in tokens if len(t) > 0]

    def tokenize_corpus(self, corpus: List[str],
                        remove_stopwords: bool = True) -> List[List[str]]:
        """Tokenize ทั้ง corpus"""
        return [self.tokenize(doc, remove_stopwords) for doc in corpus]


# =============================================================================
# SECTION 2: Legal Entity Extractor
# =============================================================================

@dataclass
class LegalEntity:
    """Legal Entity ที่สกัดได้จากข้อความ"""
    entity_type: str
    value: str
    context: str
    position: int
    law_reference: Optional[str] = None


class ThaiIPEntityExtractor:
    """
    สกัด Legal Entities จากข้อความกฎหมาย IP ไทย

    Entity Types:
    - STATUTE   : มาตราของกฎหมาย
    - PENALTY   : โทษ (จำคุก, ปรับ)
    - ACTION    : การกระทำที่ละเมิด
    - PARTY     : คู่ความ (ผู้ทรงสิทธิ, จำเลย)
    - IP_TYPE   : ประเภท IP (สิทธิบัตร, ลิขสิทธิ์)
    """

    PATTERNS = {
        "STATUTE": [
            r'มาตรา\s*[๐-๙\d]+(?:\s*(?:และ|ถึง|,)\s*มาตรา\s*[๐-๙\d]+)*',
            r'พ\.ร\.บ\.\s*[\w\s]+พ\.ศ\.\s*[๐-๙\d]+',
        ],
        "PENALTY_JAIL": [
            r'จำคุก(?:ไม่เกิน|ตั้งแต่)?\s*[\w\s]+(?:ปี|เดือน)',
            r'โทษจำคุก\s*[\w\s]+(?:ปี|เดือน)',
        ],
        "PENALTY_FINE": [
            r'ปรับ(?:ไม่เกิน|ตั้งแต่)?\s*[\w\s]+บาท',
            r'ค่าปรับ\s*[\w\s]+บาท',
        ],
        "IP_TYPE": [
            r'สิทธิบัตร(?:การประดิษฐ์|การออกแบบผลิตภัณฑ์)?',
            r'ลิขสิทธิ์',
            r'เครื่องหมายการค้า',
            r'ความลับทางการค้า',
            r'ทรัพย์สินทางปัญญา',
        ],
        "INFRINGEMENT_ACTION": [
            r'ละเมิด(?:สิทธิ(?:บัตร)?|ลิขสิทธิ์)?',
            r'ผลิต(?:สินค้า)?(?:ปลอม|เลียนแบบ)',
            r'นำเข้า(?:สินค้าปลอม)?',
            r'ทำซ้ำ(?:หรือดัดแปลง)?',
            r'เผยแพร่(?:ต่อสาธารณชน)?',
            r'จำหน่าย(?:สินค้าละเมิด)?',
        ],
        "PARTY": [
            r'ผู้ทรงสิทธิ(?:บัตร)?',
            r'เจ้าของสิทธิ(?:บัตร|ลิขสิทธิ์)?',
            r'ผู้เสียหาย',
            r'จำเลย',
            r'ผู้ละเมิด',
            r'บริษัท(?:จำเลย)?',
        ],
    }

    def extract(self, text: str) -> List[LegalEntity]:
        """สกัด entities ทั้งหมดจากข้อความ"""
        entities = []
        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    # สกัด context รอบๆ entity
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end].strip()

                    entities.append(LegalEntity(
                        entity_type=entity_type,
                        value=match.group().strip(),
                        context=context,
                        position=match.start(),
                        law_reference=self._find_statute(text, match.start())
                    ))

        # เรียงตาม position
        return sorted(entities, key=lambda e: e.position)

    def _find_statute(self, text: str, pos: int) -> Optional[str]:
        """หามาตราที่ใกล้ที่สุดก่อน position นี้"""
        statute_pattern = r'มาตรา\s*[๐-๙\d]+'
        matches = list(re.finditer(statute_pattern, text[:pos]))
        if matches:
            return matches[-1].group().strip()
        return None

    def extract_corpus(self, corpus: List[str]) -> List[List[LegalEntity]]:
        return [self.extract(doc) for doc in corpus]


# =============================================================================
# SECTION 3: TF-IDF Vectorizer สำหรับ Legal Text
# =============================================================================

class LegalTFIDF:
    """
    TF-IDF Vectorizer เฉพาะสำหรับข้อความกฎหมาย IP ไทย

    TF  = term frequency ในแต่ละเอกสาร
    IDF = inverse document frequency ทั้ง corpus

    สำหรับงานวิจัย: ใช้หาว่าคำไหน "สำคัญ" ในบริบท IP law
    """

    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = {}
        self.feature_names_ = []

    def fit(self, tokenized_corpus: List[List[str]]) -> 'LegalTFIDF':
        """เรียนรู้ vocabulary และ IDF จาก corpus"""
        N = len(tokenized_corpus)

        # นับจำนวนเอกสารที่คำปรากฏ (document frequency)
        df = collections.Counter()
        for doc_tokens in tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] += 1

        # เลือก top features
        top_terms = [term for term, _ in df.most_common(self.max_features)]
        self.vocabulary_ = {term: i for i, term in enumerate(top_terms)}
        self.feature_names_ = top_terms

        # คำนวณ IDF: log((N+1)/(df+1)) + 1  (smoothed)
        self.idf_ = {
            term: np.log((N + 1) / (count + 1)) + 1.0
            for term, count in df.items()
            if term in self.vocabulary_
        }
        return self

    def transform(self, tokenized_corpus: List[List[str]]) -> np.ndarray:
        """แปลง corpus เป็น TF-IDF matrix"""
        n_docs = len(tokenized_corpus)
        n_feats = len(self.vocabulary_)
        matrix = np.zeros((n_docs, n_feats))

        for i, doc_tokens in enumerate(tokenized_corpus):
            tf = collections.Counter(doc_tokens)
            total = len(doc_tokens) + 1e-10
            for term, count in tf.items():
                if term in self.vocabulary_:
                    j = self.vocabulary_[term]
                    tf_val = count / total
                    idf_val = self.idf_.get(term, 1.0)
                    matrix[i, j] = tf_val * idf_val

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit_transform(self, tokenized_corpus: List[List[str]]) -> np.ndarray:
        return self.fit(tokenized_corpus).transform(tokenized_corpus)

    def get_top_terms(self, tfidf_vector: np.ndarray,
                      n: int = 10) -> List[Tuple[str, float]]:
        """ดึง top-n terms จาก TF-IDF vector"""
        top_idx = np.argsort(tfidf_vector)[::-1][:n]
        return [(self.feature_names_[i], float(tfidf_vector[i]))
                for i in top_idx if tfidf_vector[i] > 0]


# =============================================================================
# SECTION 4: Co-occurrence Matrix (Word Embeddings เบื้องต้น)
# =============================================================================

class LegalCooccurrence:
    """
    สร้าง Co-occurrence Matrix สำหรับ Legal Terms

    แนวคิด: คำที่ปรากฏร่วมกันบ่อย มักมีความหมายเกี่ยวข้องกัน
    ใช้เป็น simple word embeddings ก่อนจะ upgrade เป็น BERT ใน W4

    window_size: จำนวนคำรอบข้างที่นับ co-occurrence
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.vocab_ = {}
        self.matrix_ = None

    def fit(self, tokenized_corpus: List[List[str]]) -> 'LegalCooccurrence':
        # สร้าง vocabulary
        all_tokens = [t for doc in tokenized_corpus for t in doc]
        freq = collections.Counter(all_tokens)
        # เอาเฉพาะคำที่ปรากฏมากกว่า 1 ครั้ง
        vocab_list = [t for t, c in freq.most_common() if c > 1]
        self.vocab_ = {t: i for i, t in enumerate(vocab_list)}
        V = len(self.vocab_)
        self.matrix_ = np.zeros((V, V), dtype=np.float32)

        # นับ co-occurrence
        for doc_tokens in tokenized_corpus:
            for center_i, center in enumerate(doc_tokens):
                if center not in self.vocab_:
                    continue
                ci = self.vocab_[center]
                # window รอบคำกลาง
                start = max(0, center_i - self.window_size)
                end = min(len(doc_tokens),
                          center_i + self.window_size + 1)
                for ctx_i in range(start, end):
                    if ctx_i == center_i:
                        continue
                    ctx = doc_tokens[ctx_i]
                    if ctx in self.vocab_:
                        cj = self.vocab_[ctx]
                        # ให้น้ำหนักตามระยะห่าง
                        distance = abs(ctx_i - center_i)
                        weight = 1.0 / distance
                        self.matrix_[ci, cj] += weight

        return self

    def most_similar(self, term: str,
                     n: int = 5) -> List[Tuple[str, float]]:
        """หาคำที่ co-occur บ่อยที่สุดกับ term ที่กำหนด"""
        if term not in self.vocab_:
            return []
        i = self.vocab_[term]
        row = self.matrix_[i].copy()
        row[i] = 0  # ไม่นับตัวเอง

        top_idx = np.argsort(row)[::-1][:n]
        idx2term = {v: k for k, v in self.vocab_.items()}
        return [(idx2term[j], float(row[j]))
                for j in top_idx if row[j] > 0]

    def similarity(self, term1: str, term2: str) -> float:
        """คำนวณ cosine similarity ระหว่างสองคำ"""
        if term1 not in self.vocab_ or term2 not in self.vocab_:
            return 0.0
        v1 = self.matrix_[self.vocab_[term1]]
        v2 = self.matrix_[self.vocab_[term2]]
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm == 0:
            return 0.0
        return float(np.dot(v1, v2) / norm)


# =============================================================================
# SECTION 5: IP Infringement Keyword Detector
# =============================================================================

class IPKeywordDetector:
    """
    ตรวจจับ keyword ที่บ่งชี้การละเมิด IP

    ใช้ weighted scoring:
    - Critical keywords (weight=3.0): คำที่บ่งชี้การละเมิดชัดเจน
    - High keywords    (weight=2.0): คำที่เกี่ยวข้องกับสิทธิ IP
    - Medium keywords  (weight=1.0): คำบริบทที่เกี่ยวข้อง
    """

    KEYWORD_WEIGHTS = {
        # Critical — บ่งชี้การละเมิดโดยตรง
        "ละเมิด":           3.0,
        "ปลอมแปลง":        3.0,
        "เลียนแบบ":        3.0,
        "ทำซ้ำ":           3.0,
        "ดัดแปลง":         2.5,
        "โดยไม่ได้รับอนุญาต": 3.0,
        "สินค้าปลอม":      3.0,

        # High — เกี่ยวข้องกับสิทธิ IP
        "สิทธิบัตร":       2.0,
        "ลิขสิทธิ์":       2.0,
        "เครื่องหมายการค้า": 2.0,
        "ผู้ทรงสิทธิ":     2.0,
        "เจ้าของสิทธิ":    2.0,
        "การประดิษฐ์":     1.5,

        # Medium — บริบทกฎหมาย
        "จำคุก":           1.0,
        "ปรับ":            1.0,
        "โทษ":             1.0,
        "ริบ":             1.0,
        "ความผิด":         1.0,
        "ดำเนินคดี":       1.5,
        "แจ้งความ":        1.5,
        "ศาล":             1.0,
    }

    def score(self, text: str) -> Dict:
        """คำนวณ IP infringement score จาก keyword"""
        found_keywords = {}
        total_score = 0.0

        for keyword, weight in self.KEYWORD_WEIGHTS.items():
            count = len(re.findall(re.escape(keyword), text))
            if count > 0:
                found_keywords[keyword] = {
                    "count":  count,
                    "weight": weight,
                    "score":  count * weight
                }
                total_score += count * weight

        # Normalize เป็น 0-1
        max_possible = sum(self.KEYWORD_WEIGHTS.values()) * 3
        normalized = min(total_score / max_possible, 1.0)

        return {
            "raw_score":        round(total_score, 4),
            "normalized_score": round(normalized, 4),
            "found_keywords":   found_keywords,
            "keyword_count":    len(found_keywords),
            "infringement_level": self._level(normalized)
        }

    def _level(self, score: float) -> str:
        if score >= 0.6:
            return "HIGH"
        if score >= 0.3:
            return "MEDIUM"
        if score >= 0.1:
            return "LOW"
        return "NONE"


# =============================================================================
# SECTION 6: Document Similarity (สำหรับเทียบสินค้ากับสิทธิบัตร)
# =============================================================================

def cosine_similarity_matrix(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    คำนวณ cosine similarity ระหว่างเอกสารทุกคู่

    ใช้เทียบว่าข้อความคดีละเมิดคล้ายกับ
    ข้อความสิทธิบัตรต้นฉบับแค่ไหน
    """
    # matrix already L2-normalized จาก LegalTFIDF
    return np.dot(tfidf_matrix, tfidf_matrix.T)


def find_most_similar_statute(query_text: str,
                              corpus: List[str],
                              tokenizer: ThaiLegalTokenizer,
                              tfidf: LegalTFIDF) -> List[Tuple[int, float, str]]:
    """หามาตราที่เกี่ยวข้องที่สุดกับ query"""
    query_tokens = tokenizer.tokenize(query_text, remove_stopwords=True)
    query_vector = tfidf.transform([query_tokens])[0]

    corpus_tokens = tokenizer.tokenize_corpus(corpus, remove_stopwords=True)
    corpus_matrix = tfidf.transform(corpus_tokens)

    similarities = np.dot(corpus_matrix, query_vector)
    top_idx = np.argsort(similarities)[::-1][:3]

    return [(int(i), float(similarities[i]),
             corpus[i][:60] + "...") for i in top_idx]


# =============================================================================
# MAIN: Workshop Demo
# =============================================================================

def print_section(title: str, char: str = "="):
    width = 62
    print(f"\n{char*width}")
    print(f"  {title}")
    print(f"{char*width}")


def run_workshop():  # การทำงานหลักของ workshop นี้ เริ่มที่นี่ครับ
    print("=" * 62)
    print("  WORKSHOP 1: Thai Legal Text Processing for IP Law")
    print("  Wng. Cdr. Weerayut Khrangklang | Physics-Governed IoT | 2026")
    print("=" * 62)


# ──────────────────────────────────────────────────────────
# STEP 1: Tokenization
# ──────────────────────────────────────────────────────────
print_section("STEP 1: Thai Legal Tokenization")

tokenizer = ThaiLegalTokenizer()

sample_text = THAI_IP_CORPUS[1]  # มาตรา 36 สิทธิบัตร
print(f"\nInput Text:\n{sample_text[:80]}...")

tokens = tokenizer.tokenize(sample_text, remove_stopwords=False)
tokens_no_stop = tokenizer.tokenize(sample_text, remove_stopwords=True)

print(f"\nAll tokens ({len(tokens)}):")
print("  " + " | ".join(tokens[:15]) + " ...")
print(f"\nTokens (stopwords removed) ({len(tokens_no_stop)}):")
print("  " + " | ".join(tokens_no_stop[:15]) + " ...")

 # ──────────────────────────────────────────────────────────
 # STEP 2: Legal Entity Extraction
 # ──────────────────────────────────────────────────────────
 print_section("STEP 2: Legal Entity Extraction")

  extractor = ThaiIPEntityExtractor()

   for i, doc in enumerate(THAI_IP_CORPUS[:4]):
        entities = extractor.extract(doc)
        if entities:
            print(f"\nDoc {i+1}: {doc[:45]}...")
            for e in entities[:3]:
                print(f"  [{e.entity_type:20}] {e.value}")

    # ──────────────────────────────────────────────────────────
    # STEP 3: TF-IDF Vectorization
    # ──────────────────────────────────────────────────────────
    print_section("STEP 3: TF-IDF Vectorization")

    corpus_tokens = tokenizer.tokenize_corpus(THAI_IP_CORPUS,
                                              remove_stopwords=True)
    tfidf = LegalTFIDF(max_features=80)
    tfidf_matrix = tfidf.fit_transform(corpus_tokens)

    print(f"\nCorpus size    : {len(THAI_IP_CORPUS)} documents")
    print(f"Vocabulary size: {len(tfidf.vocabulary_)} terms")
    print(f"TF-IDF matrix  : {tfidf_matrix.shape}")

    print("\nTop TF-IDF terms for 'มาตรา ๓๖ สิทธิบัตร' (Doc 2):")
    top_terms = tfidf.get_top_terms(tfidf_matrix[1], n=8)
    for term, score in top_terms:
        bar = "█" * int(score * 30)
        print(f"  {term:20} {bar} {score:.4f}")

    # ──────────────────────────────────────────────────────────
    # STEP 4: Co-occurrence Matrix
    # ──────────────────────────────────────────────────────────
    print_section("STEP 4: Co-occurrence Word Embeddings")

    cooc = LegalCooccurrence(window_size=3)
    cooc.fit(corpus_tokens)

    target_terms = ["สิทธิบัตร", "ละเมิด", "จำคุก"]
    for term in target_terms:
        similar = cooc.most_similar(term, n=4)
        if similar:
            print(f"\nคำที่เกี่ยวข้องกับ '{term}':")
            for related, score in similar:
                print(f"  → {related:20} (co-occurrence score: {score:.2f})")

    # Similarity matrix
    print("\nSimilarity ระหว่างคำสำคัญ:")
    pairs = [
        ("สิทธิบัตร", "ลิขสิทธิ์"),
        ("ละเมิด",    "จำคุก"),
        ("สิทธิบัตร", "ละเมิด"),
    ]
    for t1, t2 in pairs:
        sim = cooc.similarity(t1, t2)
        print(f"  sim('{t1}', '{t2}') = {sim:.4f}")

    # ──────────────────────────────────────────────────────────
    # STEP 5: IP Infringement Keyword Detection
    # ──────────────────────────────────────────────────────────
    print_section("STEP 5: IP Infringement Keyword Detection")

    detector = IPKeywordDetector()
    test_cases = [
        ("📋 คดีละเมิดสิทธิบัตร",
         "บริษัทจำเลยผลิตสินค้าปลอมแปลงและเลียนแบบโดยไม่ได้รับอนุญาต "
         "จากเจ้าของสิทธิบัตร ทำให้ผู้เสียหายได้รับความเสียหาย"),
        ("📋 ข้อความกฎหมายทั่วไป",
         "พระราชบัญญัติฉบับนี้มีผลบังคับใช้ตั้งแต่วันประกาศในราชกิจจานุเบกษา"),
        ("📋 คดีละเมิดลิขสิทธิ์",
         "จำเลยทำซ้ำและเผยแพร่งานอันมีลิขสิทธิ์โดยไม่ได้รับอนุญาต "
         "ต้องระวางโทษจำคุกและปรับ ดำเนินคดีกับผู้ละเมิด"),
    ]

    for label, text in test_cases:
        result = detector.score(text)
        bar = "█" * int(result["normalized_score"] * 20)
        print(f"\n  {label}")
        print(f"  Score: {bar} {result['normalized_score']:.3f} "
              f"[{result['infringement_level']}]")
        if result["found_keywords"]:
            top_kw = sorted(result["found_keywords"].items(),
                            key=lambda x: x[1]["score"], reverse=True)[:3]
            print(f"  Keywords: {', '.join(k for k, _ in top_kw)}")

    # ──────────────────────────────────────────────────────────
    # STEP 6: Document Similarity — หามาตราที่เกี่ยวข้อง
    # ──────────────────────────────────────────────────────────
    print_section("STEP 6: Statute Retrieval — หามาตราที่เกี่ยวข้อง")

    query = ("ผู้ผลิตสินค้าเลียนแบบโดยไม่ได้รับอนุญาต "
             "นำเข้าและจำหน่ายสินค้าละเมิดสิทธิบัตร")

    print(f"\nQuery: {query}")
    print("\nมาตราที่เกี่ยวข้องที่สุด:")

    results = find_most_similar_statute(
        query, THAI_IP_CORPUS, tokenizer, tfidf)
    for rank, (idx, sim, preview) in enumerate(results, 1):
        print(f"\n  #{rank} Similarity={sim:.4f}")
        print(f"     {preview}")

    # ──────────────────────────────────────────────────────────
    # STEP 7: เชื่อมกับ IoT Pipeline
    # ──────────────────────────────────────────────────────────
    print_section("STEP 7: Integration กับ IP Detection Pipeline")

    print("""
  การเชื่อมต่อกับ Pipeline ที่สร้างไว้:

  [IoT Sensors] → [Physics Gate (W17/PINNs)]
       ↓
  [Sensor Report Text] ← ป้อนเข้า W1 นี้
       ↓
  ThaiLegalTokenizer.tokenize(sensor_report)
       ↓
  ThaiIPEntityExtractor.extract()
       ↓
  IPKeywordDetector.score()
       ↓
  find_most_similar_statute() → มาตราที่ละเมิด
       ↓
  [LQM Scoring Layer] → Evidence Report
    """)

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print_section("WORKSHOP 1 SUMMARY", "▓")
    print("""
  Component             Status    Next Workshop
  ──────────────────────────────────────────────
  Thai Tokenizer        ✅ Done   W2: LSTM Baseline
  Entity Extractor      ✅ Done   W4: Fine-tune BERT
  TF-IDF Vectorizer     ✅ Done   W3: Transformer
  Co-occurrence Matrix  ✅ Done   W4: BERT Embeddings
  Keyword Detector      ✅ Done   W7: Statistical Analysis
  Statute Retrieval     ✅ Done   W12: Knowledge Graph
  Pipeline Integration  ✅ Done   W17: PINNs + NLP

  ─────────────────────────────────────────────────
  ในสภาพแวดล้อมจริง ให้แทน tokenizer ด้วย:
    pip install pythainlp
    from pythainlp.tokenize import word_tokenize
    tokens = word_tokenize(text, engine='newmm')
  ─────────────────────────────────────────────────
    """)


if __name__ == "__main__":
    run_workshop()
