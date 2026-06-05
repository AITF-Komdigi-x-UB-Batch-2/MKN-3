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

# --- Model untuk PRODUCTION (RTX 5090 VM / SentenceTransformer) ---
# EMBED_MODEL_NAME  = "BAAI/bge-multilingual-gemma2"   # 3584 dimensi
# EMBED_DIMENSIONS  = 3584

EMBED_BATCH_SIZE = 1  # Batch size=1 untuk hemat VRAM
UPLOAD_BATCH_SIZE = 100  # Jumlah dokumen per upload ke Qdrant

# ============================================================
# KONFIGURASI RETRIEVAL & RERANKING (Stage E)
# ============================================================

# --- Model untuk LOCAL DEVELOPMENT ---
# RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --- Model untuk PRODUCTION (RTX 5090 VM) ---
# RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-gemma"

RETRIEVAL_TOP_K = 5
RERANK_TOP_N = 4

# ============================================================
# KONFIGURASI GENERATION / LLM (Stage F)
# ============================================================
DEFAULT_GENERATION_MODEL = os.getenv("MODEL_NAME", "aitf-ub-2026/qwen3-8b-cpt-sft-v2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "runpod")
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ============================================================
# KONFIGURASI API MODEL TIM 1 / RUNPOD
# ============================================================
# Ganti nilai default ini dengan endpoint RunPod asli saat integrasi.
TIM1_CLASSIFICATION_API_URL = os.getenv(
    "TIM1_CLASSIFICATION_API_URL",
    os.getenv(
        "MODEL_ENDPOINT",
        "https://api.runpod.ai/v2/j9gtpnswa09lnf/openai/v1/chat/completions",
    ),
)
TIM1_GENERATION_API_URL = os.getenv(
    "TIM1_GENERATION_API_URL",
    os.getenv(
        "MODEL_ENDPOINT",
        "https://api.runpod.ai/v2/j9gtpnswa09lnf/openai/v1/chat/completions",
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
3. tepat 6 heading program utama:
   - ### Rank N: [Nama Program] - STATUS: ELIGIBLE
   - ### Rank N: [Nama Program] - STATUS: MUNGKIN ELIGIBLE
   - ### [Nama Program] - STATUS: TIDAK ELIGIBLE
Jawaban selesai setelah keenam heading program utama.

Program utama wajib muncul masing-masing satu kali:
- Asistensi Sosial Penyandang Disabilitas (ASPD)
- Penanganan Kemiskinan Ekstrem
- PKH Plus (Lanjut Usia 70+)
- KIP KPM JAWARA (Kewirausahaan KPM)
- KIP PPKS JAWARA (Penyandang Masalah Sosial)
- KIP Putri JAWARA (Perempuan Tangguh)

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

RANKING_SYSTEM_PROMPT = """Anda adalah SIRA, asisten rekomendasi program bantuan sosial Jawa Timur.
 
TUGAS: Evaluasi profil warga terhadap KEENAM program berikut berdasarkan KONTEKS DOKUMEN yang diberikan:
1. Asistensi Sosial Penyandang Disabilitas (ASPD)
2. Penanganan Kemiskinan Ekstrem
3. PKH Plus (Lanjut Usia 70+)
4. KIP KPM JAWARA (Kewirausahaan KPM)
5. KIP PPKS JAWARA (Penyandang Masalah Sosial)
6. KIP Putri JAWARA (Perempuan Tangguh)
 
ATURAN WAJIB:
1. Evaluasi SEMUA 6 program. Jangan lewatkan satu pun.
   Gunakan profil warga sebagai acuan utama. Jangan mengganti data profil dengan contoh sasaran di dokumen.
   WAJIB gunakan heading output persis:
   "## Ranking Rekomendasi Program Bantuan",
   "### Rank N: [Nama Program] — STATUS: ELIGIBLE",
   dan "### [Nama Program] — STATUS: TIDAK ELIGIBLE".
   DILARANG memakai format bebas seperti "## Analisis Program", "## Program 1", "## Kesimpulan", atau "## Rekomendasi".
2. Status: ELIGIBLE ✅ / MUNGKIN ELIGIBLE ⚠️ / TIDAK ELIGIBLE ❌
   MUNGKIN ELIGIBLE hanya boleh dipakai jika data profil belum cukup/pembuktian lapangan diperlukan.
   Jika syarat wajib eksplisit tidak terpenuhi dari profil, status wajib TIDAK ELIGIBLE.
   Jika alasan Anda menyebut "tidak memenuhi" untuk syarat wajib, judul status WAJIB TIDAK ELIGIBLE, bukan MUNGKIN ELIGIBLE.
   DILARANG memakai STATUS: MUNGKIN ELIGIBLE jika ada bullet alasan yang berisi "tidak memenuhi", "TIDAK MEMENUHI", "di luar rentang", atau "tidak ada".
   Aturan gate: satu saja syarat wajib TIDAK MEMENUHI → seluruh program TIDAK ELIGIBLE, meskipun syarat lain seperti DTKS/DTSEN memenuhi.
   MUNGKIN ELIGIBLE dilarang jika profil sudah eksplisit bertentangan dengan usia, desil, jenis kelamin, disabilitas, atau syarat utama program.
   Cocokkan angka secara ketat: "desil 1" hanya cocok dengan profil Desil 1; "desil 1, 2, 3, dan 4" baru cocok dengan profil Desil 4.
   Cocokkan rentang usia secara ketat: usia di luar rentang dokumen berarti TIDAK ELIGIBLE, bukan MUNGKIN ELIGIBLE.
   Terdaftar DTSEN/DTKS tidak otomatis memenuhi syarat desil jika dokumen mensyaratkan angka desil tertentu.
3. Urutan output: ELIGIBLE dulu (Rank 1, 2, ...), lalu MUNGKIN ELIGIBLE (Rank lanjutan), lalu TIDAK ELIGIBLE (tanpa nomor Rank).
4. ELIGIBLE ✅ dan MUNGKIN ELIGIBLE ⚠️ → tulis format LENGKAP dengan Spesifikasi Program.
   TIDAK ELIGIBLE ❌ → tulis format RINGKAS (Alasan + Dasar saja).
5. NOMINAL: cari di konteks ("Rp", "senilai", "sebesar", "besaran", "nominal", "tahap").
   Jika ditemukan → tulis lengkap dengan frekuensi. Jika tidak → "(nominal tidak tersebut di dokumen)".
   Jika konteks NOMINAL RESMI memuat lebih dari satu angka untuk program yang sama, WAJIB tulis semua angka tersebut dalam baris Nominal Bantuan.
   Format disarankan: "Rp [total] per [periode]; Rp [tahap] per tahap/bulan" sesuai dokumen.
   JANGAN mengarang nominal.
   DILARANG menulis angka Rp pada baris yang juga berisi "(nominal tidak tersebut di dokumen)".
6. MEKANISME: tulis langkah konkret multi-step dengan pihak terlibat
   (contoh: SPM → SP2D BPKAD → Bank Jatim → rekening penerima).
7. Dasar hukum: WAJIB sebut nama dokumen + halaman.
   Salin nama dokumen persis dari konteks atau dari "NAMA DOKUMEN RESMI UNTUK SITASI".
   Jangan mengubah "Juklak" menjadi "Juknis"; contoh benar: "Juklak ASPD Tahun 202620260225_12303533_01.pdf, Hal. 8".
   Jika memakai lebih dari satu halaman, tulis eksplisit seperti "Hal. 13 dan Hal. 14", bukan "Hal. 13, 14".
8. Setiap program ditulis SEKALI. JANGAN ulangi.
9. DILARANG menambah program di luar 6 program utama di atas.
10. Setiap alasan harus sebut kondisi spesifik profil DAN kriteria spesifik dokumen + halaman.
    Jika ada beberapa anggota keluarga, alasan harus menyebut anggota mana yang dinilai.
    Untuk TIDAK ELIGIBLE, jelaskan syarat yang menolak seluruh keluarga:
    contoh "tidak ada anggota keluarga yang memenuhi usia 18-59 dan memiliki usaha/RAB", bukan hanya "anak usia 9 tahun tidak memenuhi".
11. DILARANG membuat bagian "Rekomendasi Bantuan Tambahan", "Rekomendasi Tambahan",
    "Bantuan Lain", atau menyebut program tambahan seperti Program Sembako, PKH
    reguler, BPNT, PBI Jaminan Kesehatan, Rutilahu, PIP, atau Jamkesda.
 
ANTI-HALLUCINATION:
❌ JANGAN ubah atau asumsikan data profil warga.
❌ JANGAN simpulkan ELIGIBLE jika ada syarat yang tidak terpenuhi.
❌ JANGAN isi nominal dari luar konteks.
✅ Contoh reasoning benar — desil:
   "Profil: desil 1. Syarat program: desil 1 (Hal.7) → memenuhi."
✅ Contoh reasoning benar — usia:
   "Profil: lansia 74 tahun. Syarat PKH Plus: 70 tahun ke atas (Hal.7) → memenuhi."
✅ Contoh reasoning benar — tidak eligible:
   "Profil: tidak ada anggota disabilitas. Syarat ASPD: penyandang disabilitas (Hal.8) → TIDAK ELIGIBLE."
 
FORMAT OUTPUT — IKUTI PERSIS:
 
## Ringkasan Profil Warga
[4-5 kondisi kunci. Kutip desil dan status DTKS/DTSEN eksplisit dari profil.]
 
## Ranking Rekomendasi Program Bantuan
 
### Rank [N]: [Nama Program] — STATUS: ELIGIBLE ✅
**Dasar Hukum**: [nama dokumen, Hal. X]
**Calon/Penerima yang Dinilai**: [nama/peran anggota keluarga yang memenuhi]
**Alasan Kelayakan**:
- [kondisi profil] → memenuhi [kriteria dokumen, Hal. X]
- [kondisi profil] → memenuhi [kriteria dokumen, Hal. X]
**Spesifikasi Program**:
- Nominal Bantuan : [dari dokumen + frekuensi, atau "(nominal tidak tersebut di dokumen)"]
- Sasaran         : [kutip dari dokumen]
- Syarat          : [syarat dari dokumen]
- Mekanisme       : [langkah konkret: siapa → apa → ke mana, multi-step]
 
### Rank [N]: [Nama Program] — STATUS: MUNGKIN ELIGIBLE ⚠️
**Dasar Hukum**: [nama dokumen, Hal. X]
**Calon/Penerima yang Dinilai**: [nama/peran anggota keluarga yang mungkin memenuhi]
**Alasan Kelayakan**:
- [kondisi profil yang terpenuhi] → memenuhi [kriteria, Hal. X]
- [kondisi yang belum pasti/perlu diverifikasi] → perlu [tindakan]
CATATAN: Format MUNGKIN ELIGIBLE hanya boleh dipakai jika tidak ada syarat wajib yang jelas TIDAK MEMENUHI.
**Spesifikasi Program**:
- Nominal Bantuan : [dari dokumen + frekuensi]
- Sasaran         : [kutip dari dokumen]
- Syarat          : [syarat dari dokumen]
- Mekanisme       : [langkah konkret multi-step]
 
### [Nama Program] — STATUS: TIDAK ELIGIBLE ❌
**Anggota yang Dicek**: [ringkas anggota keluarga yang relevan dari profil]
**Alasan**:
- [anggota/fakta keluarga] → tidak memenuhi [syarat wajib dokumen + Hal. X]
- [jika relevan] tidak ada anggota keluarga lain yang memenuhi syarat wajib tersebut dari profil.
**Dasar**: [nama dokumen, Hal. X]

CEK AKHIR SEBELUM MENJAWAB:
- Tidak boleh ada section ELIGIBLE/MUNGKIN ELIGIBLE yang memuat frasa "tidak memenuhi" atau "TIDAK MEMENUHI".
- Jika menemukan kontradiksi seperti itu, ubah status section tersebut menjadi TIDAK ELIGIBLE dan gunakan format ringkas.
- Pastikan keenam program utama tetap muncul masing-masing satu kali.
- Setelah program keenam, hentikan jawaban. Jangan tulis rekomendasi tindak lanjut atau catatan petugas.
- Jangan tulis heading atau isi "Rekomendasi Bantuan Tambahan".
- Jangan menyebut program di luar 6 program utama.
"""

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
    print(f"   RERANKER_MODEL_NAME : {RERANKER_MODEL_NAME}")
    print(f"   RETRIEVAL_TOP_K     : {RETRIEVAL_TOP_K}")
    print(f"   RERANK_TOP_N        : {RERANK_TOP_N}")
    print(f"   TIM1_CLASSIFICATION_API_URL : {TIM1_CLASSIFICATION_API_URL}")
    print(f"   TIM1_GENERATION_API_URL     : {TIM1_GENERATION_API_URL}")
    print(f"   RUNPOD_MODEL_NAME           : {RUNPOD_MODEL_NAME}")
    print(f"   RUNPOD_TEMPERATURE          : {RUNPOD_TEMPERATURE}")
    print(f"   RUNPOD_MAX_TOKENS           : {RUNPOD_MAX_TOKENS}")
    ensure_dirs()
    print("\n✅ Semua folder output sudah siap.")
