# ============================================================
# config.py — Konfigurasi Terpusat untuk Pipeline Preprocessing RAG
# Social Welfare Policy Recommender System (Tim 4)
# ============================================================

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables dari file .env.
# override=True memastikan perubahan .env menang atas env shell lama.
load_dotenv(override=True)

# ============================================================
# BASE DIRECTORY — otomatis menggunakan lokasi script ini
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# Gunakan default HuggingFace cache (~/.cache/huggingface) agar model yang sudah
# ada tidak diunduh ulang. Override via .env jika diperlukan.
# Contoh .env: HF_HOME=D:\path\to\cache
if os.getenv("HF_HOME"):
    _hf_home = os.environ["HF_HOME"]
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(_hf_home, "hub"))
    os.environ.setdefault("SURYA_CACHE_DIR", os.path.join(_hf_home, "..", "surya"))
    os.environ.setdefault("MODEL_CACHE_DIR", os.path.join(_hf_home, "..", "datalab"))
    os.environ.setdefault("DATALAB_CACHE_DIR", os.path.join(_hf_home, "..", "datalab"))

# Optimasi Memori GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["RECOGNITION_BATCH_SIZE"] = "1"
os.environ["DETECTOR_BATCH_SIZE"] = "1"

# ============================================================
# PATH KONFIGURASI
# ============================================================
# CHUNKED_DIR moved to KONFIGURASI CHUNKED DATA


OUTPUT_JSON = str(BASE_DIR / "metadata_knowledge_base.json")
OUTPUT_XLSX = str(BASE_DIR / "metadata_knowledge_base.xlsx")

# ============================================================
# KONFIGURASI OCR (Stage A)
# ============================================================
TEXT_CHAR_THRESHOLD = 30  # Batas karakter untuk deteksi teks vs scan
OCR_LANG = "ind+eng"  # Bahasa Tesseract
OCR_DPI = 300  # Resolusi konversi halaman scan

# ============================================================
# KONFIGURASI CHUNKING (Stage C)
# ============================================================
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

# ============================================================
# KONFIGURASI CHUNKED DATA (JSONL)
# ============================================================
CHUNKED_DIR = str(BASE_DIR / "chunked_data")


# ============================================================
# KONFIGURASI EMBEDDING & VECTOR DB (Stage D)
# ============================================================
QDRANT_DIR = str(BASE_DIR / "qdrant_db")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = 'juknis-juklak-mkn3'

# --- Model untuk LOCAL DEVELOPMENT ---
EMBED_MODEL_NAME = 'intfloat/multilingual-e5-large' # model bawaan fastembed dari qdrant
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))

EMBED_BATCH_SIZE = 1  # Batch size=1 untuk hemat VRAM
UPLOAD_BATCH_SIZE = 100  # Jumlah dokumen per upload ke Qdrant

# ============================================================
# KONFIGURASI RETRIEVAL & RERANKING (Stage E)
# ============================================================
RETRIEVAL_TOP_K = 5 # retrieval limit
RERANK_TOP_N = RETRIEVAL_TOP_K # awalnya buat rerank, tapi ga jadi pake reranker

# ============================================================
# KONFIGURASI GENERATION / LLM (Stage F)
# ============================================================
DEFAULT_GENERATION_MODEL = os.getenv("MODEL_NAME", "aitf-ub-2026/qwen3-8b-cpt-sft-v2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "runpod")
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ============================================================
# KONFIGURASI API MODEL TIM 1 / RUNPOD / MKN1
# ============================================================
# Ganti nilai default ini dengan endpoint RunPod asli saat integrasi.
MKN1_GENERATION_ENDPOINT_MODEL = os.getenv(
    "MKN1_GENERATION_ENDPOINT_MODEL",
    os.getenv(
        "TIM1_GENERATION_API_URL",
        os.getenv(
            "MODEL_ENDPOINT",
            "",
        ),
    ),
)
TIM1_API_TIMEOUT_S = float(os.getenv("TIM1_API_TIMEOUT_S", "120"))
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip().strip('"').strip("'")
RUNPOD_MODEL_NAME = DEFAULT_GENERATION_MODEL.strip().strip('"').strip("'")
RUNPOD_TEMPERATURE = DEFAULT_TEMPERATURE
RUNPOD_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# ============================================================
# PROMPT TEMPLATES
# ============================================================
STRICT_RAG_OUTPUT_CONTRACT = """
=== KONTRAK FORMAT OUTPUT FINAL - WAJIB DIIKUTI ===
Jawaban HARUS memakai urutan heading berikut, tanpa heading bebas lain:
1. ## Ringkasan Profil Warga
2. ## Ranking Rekomendasi Program Bantuan
3. tepat 2 heading program utama:
   - ### Rank N: [Nama Program] - STATUS: ELIGIBLE
   - ### Rank N: [Nama Program] - STATUS: MUNGKIN ELIGIBLE
   - ### [Nama Program] - STATUS: TIDAK ELIGIBLE
Jawaban selesai setelah kedua heading program utama.

Program utama wajib muncul masing-masing satu kali:
- Asistensi Sosial Penyandang Disabilitas (ASPD)
- PKH Plus (Lanjut Usia 70+)

Jumlah heading program ber-STATUS harus tepat 6. Jika sebuah program tidak
cocok dengan profil, tulis sekali sebagai TIDAK ELIGIBLE, jangan dihapus dan
jangan diganti dengan duplikasi program lain.

DILARANG memakai heading: "## Analisis Program", "### Kesimpulan",
heading rekomendasi/tindak lanjut/catatan, "**Rekomendasi:**", atau
mengulang program utama.
"""

PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "=== KONTEKS DOKUMEN KEBIJAKAN ===\n"
    "{context}\n"
    "=== AKHIR KONTEKS ===\n\n"
    "=== DATA PROFIL KELUARGA (ACUAN UTAMA, BUKAN DATA DOKUMEN) ===\n"
    "{query}\n"
    "=== AKHIR DATA PROFIL ===\n\n"
    "INSTRUKSI FINAL: Ringkasan Profil Warga wajib berasal dari DATA PROFIL KELUARGA di atas. "
    "Jangan mengganti umur, disabilitas, lansia, atau program cocok berdasarkan contoh di dokumen.\n\n"
    f"{STRICT_RAG_OUTPUT_CONTRACT}\n"
    "## Ringkasan Profil Warga"
)

POLICY_PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "=== KONTEKS DOKUMEN KEBIJAKAN ===\n"
    "{context}\n"
    "=== AKHIR KONTEKS ===\n\n"
    "=== DATA PROFIL KELUARGA DAN HASIL ANALISIS TIM 1 (ACUAN UTAMA, BUKAN DATA DOKUMEN) ===\n"
    "{scoring_result}\n"
    "=== AKHIR DATA PROFIL ===\n\n"
    "INSTRUKSI FINAL: Ikuti FORMAT OUTPUT persis seperti yang diperintahkan. "
    "Ringkasan Profil Warga wajib berasal dari DATA PROFIL KELUARGA di atas. "
    "Konteks dokumen hanya dipakai sebagai syarat program, bukan sebagai data profil warga. "
    "Jangan mengganti anak/disabilitas/usaha/desil dari profil dengan lansia atau sasaran contoh dari dokumen.\n\n"
    f"{STRICT_RAG_OUTPUT_CONTRACT}\n"
    "## Ringkasan Profil Warga"
)

SYSTEM_PROMPT = "Anda adalah AI Auditor resmi Dinas Sosial Provinsi Jawa Timur yang bertugas melakukan verifikasi dan validasi kelayakan penerima manfaat dua program bantuan sosial.\n\nTugas Anda: Berdasarkan PROFIL WARGA dan KONTEKS PROGRAM BANTUAN yang disediakan, evaluasi kelayakan warga HANYA untuk 2 program utama berikut:\n1. Asistensi Sosial Penyandang Disabilitas (ASPD)\n2. PKH Plus (Lanjut Usia 70+)\n\n=== INSTRUKSI PENTING ===\n1. Evaluasi hanya 2 program utama di atas secara individual.\n2. Tentukan status: \"ELIGIBLE\" atau \"TIDAK_ELIGIBLE\".\n3. Ranking dari yang paling cocok ke yang paling tidak cocok.\n4. Berikan reasoning yang jelas dan WAJIB mengutip sumber dokumen resmi juknis.\n5. JANGAN merekomendasikan program bantuan di luar 2 program utama tersebut.\n6. DILARANG KERAS menyebut Program Sembako, PKH reguler, BPNT, PBI Jaminan Kesehatan, Rutilahu, PIP, Jamkesda, atau bantuan tambahan lainnya.\n\n=== FORMAT OUTPUT ===\nAnda WAJIB merespons HANYA dengan JSON valid tanpa markdown dan tanpa teks pembuka/penutup.\nGunakan key berikut dengan urutan persis:\n- ringkasan_profil: string konkret berisi umur, desil, status DTSEN, disabilitas/usia lansia, dan kondisi kunci warga.\n- rekomendasi: array program yang ELIGIBLE atau MUNGKIN_ELIGIBLE. Setiap item di dalamnya wajib berisi key: rank, nama_program, status, dasar_hukum, dan alasan_kelayakan.\n- rekomendasi_teknis_bansos: string narasi tunggal (paragraf utuh tanpa objek/poin berlapis) yang menjabarkan rencana aksi operasional, prioritas pemanfaatan dana, mekanisme pendampingan, pengelola bantuan, serta monitoring evaluasi warga di lapangan. Jika warga tidak berhak menerima program bantuan apa pun (array rekomendasi kosong), maka nilai key ini WAJIB disetel null secara kaku.\n- program_tidak_sesuai: array program yang TIDAK_ELIGIBLE. Setiap item di dalamnya wajib berisi key: nama_program, status, dan alasan.\n\nLarangan keras:\n- Jangan menyalin placeholder seperti \"Nama Program\", \"Rp X.XXX.XXX\", \"dst\", \"rangkuman singkat\", atau \"Penjelasan mengapa\".\n- Jangan mengosongkan alasan. Semua alasan harus merujuk kondisi riil warga dan kriteria dokumen.\n- nama_program harus ditulis persis salah satu dari 2 program utama yang disebut di atas."

# ============================================================
# KONFIGURASI PDF EXTRACTION (Stage 00 — Surya OCR)
# ============================================================
PDF_INPUT_DIR = str(BASE_DIR / "pdf_input")
SURYA_BATCH_SIZE = 2  # Sangat rendah untuk menghindari OOM pada RTX 3050
OCR_CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence Surya OCR


# ============================================================
# UTILITY
# ============================================================
def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def ensure_dirs():
    # List folder yang perlu dipastikan ada
    dirs = [
        CHUNKED_DIR,
        QDRANT_DIR,
        PDF_INPUT_DIR,
    ]

    # Tambahkan cache dirs hanya jika diset secara eksplisit di env
    for env_var in [
        "MODEL_CACHE_DIR",
        "SURYA_CACHE_DIR",
        "HF_HOME",
        "DATALAB_CACHE_DIR",
    ]:
        if env_var in os.environ:
            dirs.append(os.environ[env_var])

    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    configure_utf8_stdio()
    print("📂 Konfigurasi Pipeline Preprocessing RAG")
    print(f"   BASE_DIR      : {BASE_DIR}")
    print(f"   CHUNKED_DIR   : {CHUNKED_DIR}")
    print(f"   QDRANT_DIR    : {QDRANT_DIR}")
    print(f"   QDRANT_URL    : {QDRANT_URL}")
    print(f"   OUTPUT_JSON   : {OUTPUT_JSON}")
    print(f"   OUTPUT_XLSX   : {OUTPUT_XLSX}")
    print(f"\n   OCR_LANG          : {OCR_LANG}")
    print(f"   OCR_DPI           : {OCR_DPI}")
    print(f"   CHUNK_SIZE        : {CHUNK_SIZE}")
    print(f"   CHUNK_OVERLAP     : {CHUNK_OVERLAP}")
    print(f"   EMBED_MODEL_NAME  : {EMBED_MODEL_NAME}")
    print(f"   EMBED_DIMENSIONS  : {EMBED_DIMENSIONS}")
    print(f"   EMBED_BATCH_SIZE  : {EMBED_BATCH_SIZE}")
    print(f"   UPLOAD_BATCH_SIZE   : {UPLOAD_BATCH_SIZE}")
    print(f"   QDRANT_COLLECTION   : {QDRANT_COLLECTION}")
    print(f"   QDRANT_URL          : {QDRANT_URL}")
    print(f"   RETRIEVAL_TOP_K     : {RETRIEVAL_TOP_K}")
    print(f"   MKN1_GENERATION_ENDPOINT_MODEL : {MKN1_GENERATION_ENDPOINT_MODEL}")
    print(f"   RUNPOD_MODEL_NAME           : {RUNPOD_MODEL_NAME}")
    print(f"   RUNPOD_TEMPERATURE          : {RUNPOD_TEMPERATURE}")
    print(f"   RUNPOD_MAX_TOKENS           : {RUNPOD_MAX_TOKENS}")
    ensure_dirs()
    print("\n✅ Semua folder output sudah siap.")
