# ============================================================
# 00_pdf_to_jsonl.py — Direct PDF to JSONL Extractor
# Social Welfare Policy Recommender System (Tim 3)
#
# Menggunakan Surya OCR / PyMuPDF untuk mengekstrak teks dari PDF,
# lalu memetakan struktur hirarki (BAB/Pasal/Diktum/Ayat) langsung ke
# format JSONL yang siap digunakan oleh 04_embed_and_ingest_v2.py.
# Bypass tahapan CSV sepenuhnya.
# ============================================================

import os
import re
import sys
import json
import logging
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import urllib.request
import urllib.error

# Reuse utilities and constants from 00_pdf_extraction.py
# If possible, we would import them, but since they are in a script,
# we'll duplicate the necessary logic here for a standalone script, 
# or import what we can. We will import what is safe and duplicate the rest.

from config import (
    BASE_DIR, PDF_INPUT_DIR, CHUNKED_DIR,
    SURYA_BATCH_SIZE, OCR_CONFIDENCE_THRESHOLD,
    ensure_dirs,
)

import importlib.util

# Daftar ordinal diktum Indonesia (urut)
ORDINALS = [
    "KESATU", "KEDUA", "KETIGA", "KEEMPAT", "KELIMA",
    "KEENAM", "KETUJUH", "KEDELAPAN", "KESEMBILAN", "KESEPULUH",
    "KESEBELAS", "KEDUA BELAS", "KETIGA BELAS", "KEEMPAT BELAS",
    "KELIMA BELAS", "KEENAM BELAS", "KETUJUH BELAS",
    "KEDELAPAN BELAS", "KESEMBILAN BELAS", "KEDUA PULUH",
]

# ============================================================
# HIERARCHY LEVEL ENUM
# ============================================================

class HLevel(Enum):
    NONE    = 0
    BAB     = 1
    BAGIAN  = 2
    PASAL   = 3
    DIKTUM  = 4
    AYAT    = 5
    HURUF   = 6


# ============================================================
# REGEX PATTERNS — Hirarki Dokumen Hukum Indonesia
# ============================================================

# Build diktum alternatives dari ORDINALS — urutkan terpanjang dulu
_ordinal_alt = "|".join(sorted(ORDINALS, key=len, reverse=True))

HIERARCHY_PATTERNS = [
    (HLevel.BAB,    re.compile(
        r"^BAB\s+([IVXLCDM]+)(?:(?:\s*[–\-—]\s*|\s+)(.+))?$",
        re.IGNORECASE,
    )),
    (HLevel.BAGIAN, re.compile(
        r"^Bagian\s+(Kesatu|Kedua|Ketiga|Keempat|Kelima|Keenam|"
        r"Ketujuh|Kedelapan|Kesembilan|Kesepuluh|\w+)",
        re.IGNORECASE,
    )),
    (HLevel.PASAL,  re.compile(
        r"^Pasal\s+(\d+)\.?\s*$",
        re.IGNORECASE,
    )),
    # Diktum: Sangat fleksibel untuk menangkap "KESATU", "DIKTUM KESATU", 
    # atau bahkan "Pasal KESATU" (jika OCR salah baca label tapi ordinal benar)
    (HLevel.DIKTUM, re.compile(
        r"^(?:DIKTUM|Pasal)?\s*(" + _ordinal_alt + r")\s*:?\s*(.*)$",
        re.IGNORECASE,
    )),

    # AYAT/Angka: (1), (2), atau 1., 2. di awal baris
    (HLevel.AYAT, re.compile(
        r"^(?:\((\d+)\)|(\d+)\.)\s*(.*)",
        re.IGNORECASE,
    )),
    # HURUF: a., b., c. di awal baris
    (HLevel.HURUF, re.compile(
        r"^([a-z])\.\s*(.*)",
        re.IGNORECASE,
    )),
]

# Pattern untuk mendeteksi bagian preamble (skip, bukan data)
PREAMBLE_MARKERS = re.compile(
    r"^(Menimbang|Mengingat|MEMUTUSKAN|Menetapkan)\s*:", re.IGNORECASE,
)

# Map ordinal ke integer untuk pengecekan urutan
ORDINAL_MAP = {
    "KESATU": 1, "KEDUA": 2, "KETIGA": 3, "KEEMPAT": 4, "KELIMA": 5,
    "KEENAM": 6, "KETUJUH": 7, "KEDELAPAN": 8, "KESEMBILAN": 9, "KESEPULUH": 10,
    "KESEBELAS": 11, "KEDUA BELAS": 12, "KETIGA BELAS": 13, "KEEMPAT BELAS": 14, "KELIMA BELAS": 15,
    "KEENAM BELAS": 16, "KETUJUH BELAS": 17, "KEDELAPAN BELAS": 18, "KESEMBILAN BELAS": 19, "KEDUA PULUH": 20
}

# Noise patterns — baris yang harus di-skip
NOISE_PATTERNS = [
    re.compile(r"^-\s*[\dt]+\s*-$", re.I),                 # nomor halaman "- 3 -" atau "-t2-"
    re.compile(r"^(PRES[IT]DEN|REPU[BLEUK]+[A-Z]?\s+[IT]NDONESIA)$", re.I), # Header Kop: PRESIDEN / REPUBLIK INDONESIA + OCR typos
    re.compile(r"^SK\s+No\s+\d+.*$", re.I),                # footer surat keputusan (SK No 1234 A Repubuk...)
    re.compile(r"^BAB\s*[IVXLCDM]+\s*\.\.\.$", re.I),      # artifact BAB terpotong "BABII ..."
    re.compile(r"^Agar\s*\.\s*\.\s*\.?", re.I),            # footer "Agar ..."
    re.compile(r"jdih\.\w+\.go\.id", re.I),                # watermark JDIH
    re.compile(r"^Salinan sesuai dengan aslinya", re.I),   # footer salinan
    re.compile(r"^(ttd|Plt\.|NIP)", re.I),                 # tanda tangan
]

# Stop markers — menghentikan parsing (penutup & penjelasan)
# Berlaku untuk semua jenis dokumen hukum Indonesia
STOP_MARKERS = re.compile(
    r"^("
    r"Disahkan di\s|Ditetapkan di\s|Diundangkan di\s"
    r"|PENJELASAN\s+A\s*T\s*A\s*S"
    r"|Agar setiap orang mengetahuinya"
    r"|Salinan sesuai dengan aslinya"
    r")",
    re.IGNORECASE,
)

# Page-break artifact: baris yang diakhiri ". . ." (preview terpotong)
PAGE_BREAK_ARTIFACT = re.compile(r"\.\s*\.\s*\.\s*$")

# Intro line yang harus di-skip (bukan substansi)
INTRO_LINE = re.compile(
    r"^(Dalam\s+Undang|Dengan\s+persetujuan\s+bersama)", re.IGNORECASE,
)

# Pattern untuk mendeteksi judul dokumen — stop di "TENTANG ..."
TITLE_PATTERN = re.compile(
    r"((?:KEPUTUSAN|PERATURAN|INSTRUKSI|UNDANG[- ]UNDANG)"
    r"[^,]*?TENTANG\s+[^,;]+)",
    re.IGNORECASE | re.DOTALL,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PageText:
    """Teks hasil OCR dari satu halaman."""
    page_num: int
    lines: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class HierarchyState:
    """State machine untuk tracking posisi hirarki saat ini."""
    bab: str = ""
    bagian: str = ""
    pasal: str = ""
    diktum: str = ""
    ayat: str = ""

    def update(self, level: HLevel, value: str):
        if level == HLevel.BAB:
            self.bab = value
            self.bagian = ""
            self.pasal = ""
            self.diktum = ""
            self.ayat = ""
        elif level == HLevel.BAGIAN:
            self.bagian = value
            self.pasal = ""
            self.diktum = ""
            self.ayat = ""
        elif level == HLevel.PASAL:
            self.pasal = value
            self.diktum = ""
            self.ayat = ""
        elif level == HLevel.DIKTUM:
            self.diktum = value
            self.ayat = ""
        elif level in (HLevel.AYAT, HLevel.HURUF):
            self.ayat = value

    @property
    def bab_display(self) -> str:
        """Return BAB saja tanpa Bagian (sesuai ground truth)."""
        return self.bab if self.bab else "-"

    @property
    def pasal_display(self) -> str:
        if self.diktum:
            return self.diktum
        if self.pasal:
            return self.pasal
        return "-"

    @property
    def ayat_display(self) -> str:
        return self.ayat if self.ayat else "-"


# ============================================================
# TEXT EXTRACTION — Surya OCR
# ============================================================

def extract_text_surya(pdf_path: str) -> list[PageText]:
    """
    Ekstrak teks dari PDF menggunakan Surya OCR.
    Returns list of PageText, satu per halaman.
    """
    try:
        import pypdfium2 as pdfium
        from surya.recognition import RecognitionPredictor, FoundationPredictor
        from surya.detection import DetectionPredictor

        # PATCH: Fix SuryaDecoderConfig missing pad_token_id (common transformers bug)
        try:
            from surya.model.recognition.config import SuryaDecoderConfig
            if not hasattr(SuryaDecoderConfig, "pad_token_id"):
                SuryaDecoderConfig.pad_token_id = 2  # Default for Surya
                logger.info("🔧 Patch SuryaDecoderConfig.pad_token_id diterapkan.")
        except ImportError:
            pass
    except ImportError as e:
        logger.error(
            "❌ Library Surya OCR belum terinstall: %s\n"
            "   Jalankan: pip install surya-ocr", e,
        )
        raise

    logger.info("📄 Membuka PDF: %s", pdf_path)
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    logger.info("   Total halaman: %d", num_pages)

    # Render halaman PDF ke gambar PIL
    images = []
    for page_idx in range(num_pages):
        page = pdf[page_idx]
        # Render pada 300 DPI
        bitmap = page.render(scale=300 / 72)
        pil_image = bitmap.to_pil()
        images.append(pil_image)

    # Inisialisasi Surya predictors (API v0.17+)
    logger.info("📦 Memuat model Surya OCR ...")
    det_predictor = DetectionPredictor()
    foundation_predictor = FoundationPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    # Jalankan OCR
    logger.info("🔍 Menjalankan OCR pada %d halaman ...", num_pages)
    os.environ["RECOGNITION_BATCH_SIZE"] = str(SURYA_BATCH_SIZE)

    predictions = rec_predictor(images, det_predictor=det_predictor)

    pages = []
    for page_idx, pred in enumerate(predictions):
        lines = []
        total_conf = 0.0
        count = 0
        for text_line in pred.text_lines:
            text = text_line.text.strip()
            if text:
                lines.append(text)
                total_conf += text_line.confidence
                count += 1

        avg_conf = total_conf / count if count > 0 else 0.0
        pages.append(PageText(
            page_num=page_idx + 1,
            lines=lines,
            confidence=avg_conf,
        ))
        logger.info(
            "   Halaman %d: %d baris (conf=%.2f)",
            page_idx + 1, len(lines), avg_conf,
        )

    return pages


def extract_text_pymupdf(pdf_path: str) -> list[PageText]:
    """Fallback: ekstrak teks menggunakan PyMuPDF (cerdas mengurutkan blok)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("❌ PyMuPDF (fitz) belum terinstall: pip install PyMuPDF")
        raise

    logger.info("📄 [Fallback PyMuPDF] Membuka: %s", pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        # Menggunakan "blocks" agar urutan teks lebih terjaga (atas ke bawah)
        blocks = page.get_text("blocks")
        # Urutkan blok berdasarkan koordinat y (atas ke bawah) lalu x (kiri ke kanan)
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        lines = []
        for b in blocks:
            text = b[4].strip()
            if text:
                # Pecah blok menjadi baris-baris individu
                for line in text.split("\n"):
                    if line.strip():
                        lines.append(line.strip())
        
        pages.append(PageText(page_num=page_idx + 1, lines=lines, confidence=1.0))

    logger.info("   Total halaman: %d", len(pages))
    return pages


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_noise(line: str) -> bool:
    """Cek apakah baris termasuk noise (header/footer/watermark)."""
    for pattern in NOISE_PATTERNS:
        if pattern.search(line):
            return True
    return False


def is_diktum_line(line: str) -> bool:
    """Cek apakah baris merupakan awal diktum baru."""
    stripped = line.strip()
    for level, pattern in HIERARCHY_PATTERNS:
        if level == HLevel.DIKTUM and pattern.match(stripped):
            return True
    return False


# ============================================================
# TITLE & KATEGORI DETECTION
# ============================================================

def detect_title(pages: list[PageText]) -> str:
    """
    Deteksi judul dokumen dari halaman pertama.
    Membaca teks dari awal hingga kata "Menimbang" atau "Mengingat",
    lalu membersihkannya untuk mendapatkan judul asli secara utuh.
    """
    if not pages:
        return "Dokumen Tidak Diketahui"

    all_text = " ".join(" ".join(p.lines) for p in pages[:2])
    
    # 1. Potong sebelum Menimbang/Mengingat agar tidak membaca isi/konsideran
    cut_match = re.search(r"\b(Menimbang|Mengingat)\b", all_text, re.IGNORECASE)
    if cut_match:
        all_text = all_text[:cut_match.start()]
        
    # 2. Hapus noise umum di header dokumen
    all_text = re.sub(r"\bjdih\.\w+\.go\.id\b", " ", all_text, flags=re.I)
    all_text = re.sub(r"-\s*\d+\s*-", " ", all_text)
    # Hapus huruf s nyasar (karena scanning) atau teks salinan
    all_text = re.sub(r"\b(s|salinan)\b", " ", all_text, flags=re.I)
    # Hapus ttd penutup di header
    all_text = re.sub(r"MENTERI SOSIAL REPUBLIK INDONESIA\s*,$", "", all_text, flags=re.I)
    
    all_text = re.sub(r"\s+", " ", all_text).strip()
    all_text = all_text.rstrip(",;. ")

    # 3. Coba regex standar dulu (kalau ada format TENTANG yang rapi)
    m = TITLE_PATTERN.search(all_text)
    if m:
        return m.group(1).strip()
        
    # 4. Kalau tidak ada TENTANG, tapi ada teks KEPUTUSAN dsb
    # Kembalikan semua sisa teks di area header (karena sudah aman dipotong sebelum Menimbang)
    if len(all_text) > 30 and re.search(r"(KEPUTUSAN|PERATURAN|UNDANG)", all_text, re.I):
        return all_text
        
    # 5. Fallback akhir
    for p in pages[:2]:
        for line in p.lines:
            if len(line) > 20 and not line.startswith("-") and not is_noise(line):
                return line.strip()

    return "Dokumen Tidak Diketahui"


def title_case_bab(text: str) -> str:
    """Convert BAB title dari ALL CAPS ke Title Case."""
    m = re.match(r"(BAB\s+[IVXLCDM]+)\s*[–\-—]\s*(.*)", text)
    if not m:
        return text
    prefix = m.group(1)
    title_part = m.group(2).strip()
    small = {"dan", "atau", "di", "ke", "dari", "yang", "dalam",
             "untuk", "dengan", "pada", "serta", "atas"}
    words = title_part.split()
    result = []
    for i, w in enumerate(words):
        if i == 0:
            result.append(w.capitalize())
        elif w.lower() in small:
            result.append(w.lower())
        else:
            result.append(w.capitalize())
    return prefix + " – " + " ".join(result)


def shorten_title(title: str) -> str:
    """
    Buat versi pendek judul untuk kolom Sumber.
    Generic untuk semua jenis dokumen hukum Indonesia.
    """
    s = title
    # Bersihkan frasa seremoni/header (umum di semua dokumen hukum)
    s = re.sub(r"\s*REPUBLIK\s+INDONESIA\s*", " ", s, flags=re.I)
    s = re.sub(r"\s+dengan\s+Rahmat\s+Tuhan.*$", "", s, flags=re.I)
    s = re.sub(r"\s+(Presiden|Menteri\s+[A-Za-z\s]+)\s*$", "", s, flags=re.I)

    # Normalisasi nomor
    s = re.sub(r"NOMOR\s+", "No. ", s, flags=re.I)

    # Singkatan standar jenis dokumen hukum Indonesia
    abbreviations = [
        (r"UNDANG[\s-]+UNDANG", "UU"),
        (r"PERATURAN\s+PEMERINTAH", "PP"),
        (r"PERATURAN\s+PRESIDEN", "Perpres"),
        (r"INSTRUKSI\s+PRESIDEN", "Inpres"),
        (r"PERATURAN\s+MENTERI\b", "Permen"),
        (r"KEPUTUSAN\s+MENTERI\b", "Kepmen"),
        (r"PERATURAN\s+DAERAH", "Perda"),
        (r"PERATURAN\s+GUBERNUR", "Pergub"),
        (r"PERATURAN\s+BUPATI", "Perbup"),
        (r"PERATURAN\s+WALIKOTA", "Perwali"),
        (r"KEPUTUSAN\s+DIREKTUR\s+JENDERAL", "Kepdirjen"),
    ]
    for pattern, abbr in abbreviations:
        s = re.sub(pattern, abbr, s, flags=re.I)

    # Title-case conversion
    small = {"dan", "atau", "di", "ke", "dari", "atas", "yang",
             "dalam", "untuk", "dengan", "pada", "secara", "oleh",
             "melalui", "serta"}
    # Daftar singkatan yang harus dipertahankan
    keep_as_is = {"UU", "PP", "Perpres", "Inpres", "Permen", "Kepmen",
                  "Perda", "Pergub", "Perbup", "Perwali", "Kepdirjen"}
    words = s.split()
    result = []
    for i, w in enumerate(words):
        if w in keep_as_is:
            result.append(w)
        elif w.lower() == "tentang":
            result.append("tentang")
        elif re.match(r"^\d", w) or "/" in w:
            result.append(w)
        elif w.lower() in small and i > 0:
            result.append(w.lower())
        elif w.upper() == w and len(w) > 2:
            result.append(w.capitalize())
        else:
            result.append(w)
    short = re.sub(r"\s+", " ", " ".join(result)).strip()
    if short:
        short = short[0].upper() + short[1:]
    return short


# Daftar kategori valid untuk klasifikasi dokumen
KATEGORI_OPTIONS = [
    "Regulasi (Aturan Hukum & Payung Kebijakan)",
    "Juknis (Petunjuk Teknis & SOP)",
    "Pedoman Program (Rencana Kerja & Aksi)",
    "Data dan Laporan Analisis",
]


def infer_kategori(sumber: str) -> str:
    """
    Klasifikasi kategori dokumen menggunakan heuristik sederhana (pencarian keyword).
    Jauh lebih cepat daripada menggunakan LLM, dan cukup akurat untuk dokumen hukum Indonesia.
    """
    s = sumber.lower()
    if any(k in s for k in ["juknis", "petunjuk teknis", "sop", "prosedur"]):
        return KATEGORI_OPTIONS[1]
    if any(k in s for k in ["pedoman", "rencana", "aksi", "panduan"]):
        return KATEGORI_OPTIONS[2]
    if any(k in s for k in ["data", "laporan", "statistik", "analisis", "survei"]):
        return KATEGORI_OPTIONS[3]
    return KATEGORI_OPTIONS[0]  # Default: Regulasi


# ============================================================
# HIERARCHY PARSING
# ============================================================

def classify_line(line: str) -> tuple[HLevel, str, str]:
    """
    Klasifikasi satu baris teks ke level hirarki.
    Returns: (level, label, remaining_text)
    """
    stripped = line.strip()

    for level, pattern in HIERARCHY_PATTERNS:
        m = pattern.match(stripped)
        if m:
            groups = m.groups()
            if level == HLevel.BAB:
                label = f"BAB {groups[0]}"
                remaining = groups[1].strip() if len(groups) > 1 and groups[1] else ""
                return level, label, remaining
            elif level == HLevel.BAGIAN:
                label = f"Bagian {groups[0]}"
                return level, label, ""
            elif level == HLevel.PASAL:
                label = f"Pasal {groups[0]}"
                return level, label, ""
            elif level == HLevel.DIKTUM:
                # Pastikan label selalu diawali kata "Diktum"
                ordinal_name = groups[0].upper()
                label = f"Diktum {ordinal_name}"
                remaining = groups[1].strip() if len(groups) > 1 else ""
                # Strip leading colon dari remaining
                remaining = remaining.lstrip(": ").strip()
                return level, label, remaining
            elif level == HLevel.AYAT:
                val = groups[0] if groups[0] else groups[1]
                remaining = groups[2].strip() if len(groups) > 2 else ""
                return level, val, remaining
            elif level == HLevel.HURUF:
                val = groups[0]
                remaining = groups[1].strip() if len(groups) > 1 else ""
                return level, val, remaining

    return HLevel.NONE, "", stripped


spec_norm = importlib.util.spec_from_file_location("normalize_jsonl", os.path.join(BASE_DIR, "03_normalize_jsonl.py"))
norm_ext = importlib.util.module_from_spec(spec_norm)
sys.modules["normalize_jsonl"] = norm_ext
spec_norm.loader.exec_module(norm_ext)

expand_abbreviations = norm_ext.expand_abbreviations
normalize_text = norm_ext.normalize_text

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# File manifest untuk tracking PDF yang sudah diproses
MANIFEST_FILE = os.path.join(CHUNKED_DIR, "pdf_processed_manifest.json")

# ============================================================
# FUNGSI UTILITAS MANIFEST & KONTEKS
# ============================================================

def file_hash(filepath: str) -> str:
    """Hitung SHA-256 hash dari sebuah file untuk deteksi perubahan."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()

def load_manifest() -> dict:
    if os.path.isfile(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("⚠️ Manifest rusak (%s), akan dibangun ulang.", e)
    return {}

def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def build_konteks_lengkap(bab: str, pasal: str, ayat: str) -> str:
    parts = []
    if bab and bab != "-":
        parts.append(f"Bab {bab}")
    if pasal and pasal != "-":
        parts.append(f"Pasal/Diktum {pasal}")
    if ayat and ayat != "-":
        parts.append(f"Ayat {ayat}")
    return " | ".join(parts)


# ============================================================
# JSONL BUILDER
# ============================================================

def build_jsonl_rows(
    pages: list[PageText],
    sumber: str,
    kategori: str,
) -> list[dict]:
    """
    Iterasi semua baris OCR, deteksi hirarki, dan bangun objek JSONL.
    """
    state = HierarchyState()
    rows = []
    row_num = 0
    content_buffer = []
    in_preamble = True
    expecting_bab_title = False
    skip_next_line = False
    current_page_num = 1
    starting_page_num = 1

    def flush_buffer():
        nonlocal row_num, starting_page_num
        if not content_buffer:
            return

        text = " ".join(content_buffer).strip()
        text = re.sub(r"\s*\.\s*\.\s*\.\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        
        penutup = re.search(
            r"(?:Agar setiap orang mengetahuinya"
            r"|Disahkan di [A-Z][a-z]+"
            r"|Ditetapkan di [A-Z][a-z]+"
            r"|Diundangkan di [A-Z][a-z]+"
            r"|memerintahkan pengundangan"
            r"|Salinan sesuai dengan aslinya)",
            text, re.I,
        )
        if penutup:
            text = text[:penutup.start()].rstrip(" .,;")

        if not text or len(text) < 3:
            content_buffer.clear()
            return

        row_num += 1
        pasal_label = state.pasal_display
        
        bab_display = state.bab_display
        ayat_display = state.ayat_display
        konteks_lengkap = build_konteks_lengkap(bab_display, pasal_label, ayat_display)
        
        # Simplified and clean metadata for Vector DB
        metadata = {
            "sumber": sumber,
            "bab": bab_display if bab_display != "-" else "",
            "pasal": pasal_label if pasal_label != "-" else "",
            "ayat": ayat_display if ayat_display != "-" else "",
            "kategori": kategori,
            "page_number": starting_page_num,
        }

        # Normalize and expand abbreviations
        normalized_text = normalize_text(expand_abbreviations(text))
        
        # Semantic Chunking Prefix
        prefixed_text = f"[{sumber} | {konteks_lengkap}] {normalized_text}"

        rows.append({
            "No": str(row_num),
            "Isi / Substansi": prefixed_text,
            "_final_metadata": metadata # Disimpan untuk JSONL final
        })
        content_buffer.clear()


    def append_to_buffer(text_chunk: str):
        nonlocal starting_page_num
        if not content_buffer:
            starting_page_num = current_page_num
        content_buffer.append(text_chunk)

    def buffer_incomplete() -> bool:
        if not content_buffer:
            return False
        last = content_buffer[-1].rstrip()
        if not last:
            return False
        return last[-1] not in ".;:"

    # ======= MAIN LOOP =======
    for page in pages:
        current_page_num = page.page_num
        if page.confidence < OCR_CONFIDENCE_THRESHOLD and page.confidence > 0:
            logger.warning("⚠️ Halaman %d confidence rendah (%.2f).", page.page_num, page.confidence)

        # Pre-process lines to split embedded markers (e.g. "... perjudian. KEDELAPAN : ...")
        processed_lines = []
        for line in page.lines:
            # Pattern: [. ] (OPTIONAL_LABEL) ORDINAL [:]
            # Kita split jika ada marker di tengah baris (setelah titik atau spasi cukup lebar)
            split_pattern = r"(?<=\.)\s+(?:DIKTUM|Pasal)?\s*(" + _ordinal_alt + r")\s*:\s*"
            parts = re.split(split_pattern, line, flags=re.IGNORECASE)
            
            if len(parts) > 1:
                # Part 0: teks sebelum marker
                if parts[0].strip():
                    processed_lines.append(parts[0].strip())
                
                # Sisa parts: [ORDINAL, teks_setelah, ORDINAL, teks_setelah, ...]
                for i in range(1, len(parts), 2):
                    ordinal = parts[i]
                    after = parts[i+1] if i+1 < len(parts) else ""
                    processed_lines.append(f"{ordinal} : {after}".strip())
            else:
                processed_lines.append(line)

        for line in processed_lines:
            stripped = line.strip()

            if not stripped:
                continue

            stripped = re.sub(r"^\((\d+)[1lI|\]\}\)]\s+", r"(\1) ", stripped)

            if STOP_MARKERS.search(stripped):
                flush_buffer()
                return rows

            if is_noise(stripped):
                continue
            if PAGE_BREAK_ARTIFACT.search(stripped):
                continue
            if INTRO_LINE.match(stripped):
                continue
            if skip_next_line:
                skip_next_line = False
                continue
            if PREAMBLE_MARKERS.match(stripped):
                in_preamble = True
                continue

            is_cont = False
            if content_buffer:
                prev = content_buffer[-1].lower().strip()
                connectors = ["diktum", "pasal", "dalam", "yaitu", "meliputi",
                              "dan", "sebagaimana", "pada", "tentang", "bahwa",
                              "keputusan", "ayat"]
                if any(prev.endswith(w) for w in connectors):
                    is_cont = True

            level, label, remaining = classify_line(stripped)

            if buffer_incomplete() and level == HLevel.AYAT:
                level = HLevel.NONE

            if level in (HLevel.PASAL, HLevel.DIKTUM) and not is_cont:
                new_val_str = label.split()[-1].upper()
                new_val = ORDINAL_MAP.get(new_val_str, 0)
                curr_str = (state.pasal or state.diktum).split()[-1].upper() if (state.pasal or state.diktum) else ""
                curr_val = ORDINAL_MAP.get(curr_str, 0)
                if curr_val > 0 and new_val > 0 and new_val <= curr_val:
                    level = HLevel.NONE

            if is_cont and level in (HLevel.PASAL, HLevel.DIKTUM):
                level = HLevel.NONE

            if level in (HLevel.BAB, HLevel.BAGIAN, HLevel.PASAL, HLevel.DIKTUM):
                in_preamble = False

            if expecting_bab_title:
                if level in (HLevel.PASAL, HLevel.BAGIAN):
                    state.bab = title_case_bab(state.bab)
                    expecting_bab_title = False
                else:
                    if " – " in state.bab:
                        state.bab += " " + stripped
                    else:
                        state.bab += " – " + stripped
                    continue

            if in_preamble:
                continue

            if level == HLevel.BAB:
                flush_buffer()
                if remaining:
                    bab_val = label + " – " + remaining
                    state.update(HLevel.BAB, title_case_bab(bab_val))
                else:
                    state.update(HLevel.BAB, label)
                    expecting_bab_title = True
            elif level == HLevel.BAGIAN:
                skip_next_line = True
                continue
            elif level in (HLevel.PASAL, HLevel.DIKTUM):
                flush_buffer()
                state.update(level, label)
                if remaining:
                    append_to_buffer(remaining)
            elif level == HLevel.AYAT:
                if state.diktum:
                    append_to_buffer(stripped)
                else:
                    flush_buffer()
                    state.update(level, label)
                    if remaining:
                        append_to_buffer(remaining)
            elif level == HLevel.HURUF:
                append_to_buffer(stripped)
            else:
                clean_text = re.sub(r"^[.:\s]+", "", stripped)
                if clean_text:
                    append_to_buffer(clean_text)

    flush_buffer()
    return rows


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_pdf_to_jsonl(
    pdf_path: str,
    output_dir: str = CHUNKED_DIR,
    use_surya: bool = True,
) -> str:
    """
    Ekstrak PDF dan simpan langsung sebagai JSONL.
    """
    pdf_name = Path(pdf_path).stem

    try:
        import fitz
        doc = fitz.open(pdf_path)
        total_chars = 0
        for page in doc:
            total_chars += len(page.get_text().strip())
        is_scan = (total_chars / len(doc)) < 50 if len(doc) > 0 else True
        doc.close()
    except Exception:
        is_scan = True

    if is_scan and use_surya:
        logger.info("📸 Scan PDF terdeteksi. Menggunakan Surya OCR (Slow Path)...")
        try:
            pages = extract_text_surya(pdf_path)
        except Exception as e:
            logger.warning("⚠️ Surya OCR gagal (%s). Fallback ke PyMuPDF ...", e)
            pages = extract_text_pymupdf(pdf_path)
    else:
        logger.info("⚡ Native PDF terdeteksi. Menggunakan PyMuPDF (Fast Path)...")
        pages = extract_text_pymupdf(pdf_path)

    if not pages:
        logger.error("❌ Tidak ada halaman yang berhasil diekstrak.")
        return ""

    raw_title = detect_title(pages)
    title = shorten_title(raw_title)
    logger.info("📋 Judul terdeteksi: %s", title)

    kategori = infer_kategori(title)

    rows = build_jsonl_rows(pages, sumber=title, kategori=kategori)
    logger.info("📊 Total baris data: %d", len(rows))

    if not rows:
        logger.warning("⚠️ Tidak ada baris data yang dihasilkan dari PDF ini.")
        return ""

    # Konversi ke format JSONL final
    final_chunks = []
    for r in rows:
        metadata = r["_final_metadata"]
        
        final_chunks.append({
            "text": r["Isi / Substansi"],
            "metadata": metadata
        })

    # Simpan JSONL
    output_path = os.path.join(output_dir, f"{pdf_name}.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in final_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("💾 JSONL disimpan: %s (%d chunks)", output_path, len(final_chunks))
    return output_path


def process_all_pdfs(
    input_dir: str = PDF_INPUT_DIR,
    output_dir: str = CHUNKED_DIR,
    use_surya: bool = True,
    force: bool = False
) -> list[str]:
    """
    Proses semua file PDF di input_dir → JSONL di output_dir.
    """
    if not os.path.isdir(input_dir):
        logger.error("❌ Folder input tidak ditemukan: %s", input_dir)
        return []

    pdf_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        logger.warning("⚠️ Tidak ada file PDF di: %s", input_dir)
        return []

    manifest = load_manifest() if not force else {}
    to_process = []
    already_done = []

    for fpath in pdf_files:
        fname = os.path.basename(fpath)
        current_hash = file_hash(fpath)
        
        target_jsonl = os.path.join(output_dir, f"{Path(fpath).stem}.jsonl")

        if fname in manifest and manifest[fname].get("hash") == current_hash and os.path.exists(target_jsonl):
            already_done.append(fpath)
        else:
            to_process.append(fpath)

    logger.info("📂 Ditemukan %d file PDF. %d perlu diproses, %d sudah diproses.", 
                len(pdf_files), len(to_process), len(already_done))

    results = []
    for fpath in to_process:
        fname = os.path.basename(fpath)
        logger.info("🔄 Memproses: %s ...", fname)
        try:
            jsonl_path = process_pdf_to_jsonl(fpath, output_dir, use_surya)
            if jsonl_path:
                results.append(jsonl_path)
                
                # Update manifest
                manifest[fname] = {
                    "hash": file_hash(fpath),
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_manifest(manifest)
                
        except Exception as e:
            logger.error("❌ Gagal memproses %s: %s", fname, e)

    return results

def main():
    ensure_dirs()
    start = time.time()

    print("=" * 65)
    print("🚀 STAGE 00: PDF to JSONL Extractor")
    print(f"   Input  : {PDF_INPUT_DIR}")
    print(f"   Output : {CHUNKED_DIR}")
    print(f"   Manifest: {MANIFEST_FILE}")
    print("=" * 65)

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if pdf_path.startswith("--"):
            use_surya = "--pymupdf" not in sys.argv
            force = "--force" in sys.argv
            results = process_all_pdfs(use_surya=use_surya, force=force)
            print(f"\n📊 Total JSONL dihasilkan: {len(results)}")
        else:
            if not os.path.isfile(pdf_path):
                print(f"❌ File tidak ditemukan: {pdf_path}")
                sys.exit(1)

            use_surya = "--pymupdf" not in sys.argv
            jsonl_path = process_pdf_to_jsonl(pdf_path, use_surya=use_surya)
            if jsonl_path:
                # Update manifest for this single file
                manifest = load_manifest()
                fname = os.path.basename(pdf_path)
                manifest[fname] = {
                    "hash": file_hash(pdf_path),
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_manifest(manifest)
                print(f"\n✅ Output: {jsonl_path}")
            else:
                print("\n❌ Gagal mengekstrak PDF.")
    else:
        results = process_all_pdfs()
        print(f"\n📊 Total JSONL dihasilkan: {len(results)}")

    elapsed = time.time() - start
    print(f"\n⏱️ Waktu: {elapsed:.1f} detik")
    print("=" * 65)


if __name__ == "__main__":
    main()
