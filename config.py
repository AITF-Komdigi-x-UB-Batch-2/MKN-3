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
QDRANT_COLLECTION = 'juknis-juklak-filtered'

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
DEFAULT_GENERATION_MODEL = os.getenv("MODEL_NAME", "ub-mkn-all")
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
RUNPOD_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# ============================================================
# FEW-SHOT EXAMPLES (TOON format)
# Digunakan sebagai in-context learning untuk membimbing model
# menghasilkan output TOON yang konsisten.
# ============================================================

FEW_SHOT_USER_EXAMPLE = (
    "=== PROFIL WARGA ===\n"
    "- NIK / No. KK     : PRS_4b5389362ecd8a49225ce6cb51a267f414e96693eb6a1461c42542b469a0244f / FAM_25825cc1cb27c99e5f01836f56d13a4cf702e47fc5d633f845efaf8338869340\n"
    "- Nama             : ***ALI\n"
    "- Umur             : 77 tahun\n"
    "- Hub. Kepala KK   : Kepala keluarga\n"
    "- Status Kawin     : Kawin\n"
    "- Jml. Anggota KK  : 5 orang\n"
    "- Desil Nasional   : 2 | Status DTSEN: DTSEN AKTIF\n"
    "- Status Keberadaan: Ditemukan / Aktif\n"
    "- Bansos           : PKH, SEMBAKO\n"
    "- PBI Jaminan Kes  : Ya\n"
    "- Kondisi Gizi     : Tidak diketahui\n"
    "- Penyakit Menahun : Tidak ada\n"
    "Hambatan Fungsi:\n"
    "- Penglihatan      : Tidak mengalami kesulitan | Pendengaran: Tidak mengalami kesulitan\n"
    "- Berjalan/Tangga  : Tidak mengalami kesulitan | Tangan/Jari: Tidak mengalami kesulitan\n"
    "- Belajar/Intelek  : Tidak mengalami kesulitan | Perilaku: Tidak mengalami kesulitan\n"
    "- Bicara/Komunikasi: Tidak mengalami kesulitan | Mengurus Diri: Tidak mengalami kesulitan\n"
    "- Ingatan/Fokus    : Tidak mengalami kesulitan | Sedih/Depresi: Tidak mengalami kesulitan\n"
    "- Wilayah          : Purwantoro, Kec. Blimbing, Kota Malang, Jawa Timur\n"
    "=== AKHIR PROFIL WARGA ===\n\n"
    "=== KONTEKS DOKUMEN KEBIJAKAN DARI RETRIEVAL ===\n"
    "<dokumen_juknis>\n"
    "a) Meningkatkan taraf hidup dan kesejahteraan penerima manfaat melalui\n"
    "penerima manfaat;\n"
    "penerima manfaat dalam mengakses layanan kesehatan dan\n"
    "2. Sasaran Penerima Bantuan sosial PKH Plus\n"
    "a) Lanjut usia 70 Tahun ke atas seorang diri dan/atau yang tercatat dalam satu\n"
    "kartu keluarga dalam keluarga penerima manfaat Program Keluarga Harapan\n"
    "tingkat kesejahteraan sosial atau desil 1, 2, 3 dan 4;\n"
    "d) Dalam hal penerima manfaat dalam satu keluarga terdapat lebih dari satu\n"
    "Sesuai Surat Keputusan Gubernur Jawa Timur tentang Penerima Manfaat PKH\n\n"
    "Dalam rangka perlindungan dan jaminan sosial bagi lanjut usia, menjadi\n"
    "inklusif. Lanjut usia merupakan kelompok masyarakat yang secara sosial dan\n"
    "lanjut usia 70 tahun ke atas, Pemerintah Provinsi Jawa Timur memberikan bantuan\n"
    "prinsip 6T (tepat sasaran, tepat waktu, tepat jumlah, tepat administrasi, tepat kualitas,\n\n"
    "bantuan sosial PKH Plus kepada penerima manfaat PKH Plus berdasarkan berita\n"
    "Penerima manfaat sudah tidak menerima bantuan sosial PKH regular dari\n"
    "Status penerima manfaat yang meninggal dunia dilaporkan dengan melampirkan\n"
    "Status penerima manfaat yang sudah mampu secara ekonomi.\n"
    "Status penerima manfaat yang pindah alamat domisili di luar Provinsi Jawa Timur.\n"
    "5. Data penerima ganda\n"
    "Penerima manfaat yang duplikasi individu/keluarga yang terdaftar sebagai\n"
    "penerima manfaat bantuan sosial PKH Plus\n"
    "</dokumen_juknis>\n"
    "=== AKHIR KONTEKS DOKUMEN ===\n\n"
    "INSTRUKSI EKSEKUSI:\n"
    "1. Lakukan audit kelayakan secara objektif dengan mencocokkan kriteria pada Profil Warga terhadap aturan di Konteks Dokumen.\n"
    "2. Hasilkan output TEPAT dalam format Toon dengan empat kategori wajib: 'ringkasan_profil', 'rekomendasi', 'rekomendasi_teknis_bansos', dan 'program_tidak_sesuai'.\n"
    "3. Pada baris 'rekomendasi', Anda wajib memuat informasi: rank, dasar hukum, dan alasan kelayakan di dalam kolom Detail/Alasan.\n"
    "4. Pada baris 'program_tidak_sesuai', Anda wajib memuat informasi: alasan ketidaksesuaian di dalam kolom Detail/Alasan.\n"
    "5. Respons hanya berupa teks format Toon valid. Jangan pernah memakai tag markdown pembungkus (seperti ```toon ...), heading teks tambahan di luar struktur, atau placeholder."
)

FEW_SHOT_ASSISTANT_EXAMPLE = (
    "Hasil[4]{Kategori,Nilai/Program,Status,Detail/Alasan}:\n"
    "ringkasan_profil,Profil_Warga,-,\"Warga berusia 77 tahun dengan posisi hubungan keluarga sebagai Kepala keluarga dan status pernikahan Kawin. Secara ekonomi, status kesejahteraan berada pada desil nasional 2 dengan status keberadaan lapangan: Ditemukan / Aktif. Kondisi kesehatan mencatat riwayat gizi Tidak diketahui, status kepesertaan PBI Jaminan Kesehatan: Ya, serta indikasi penyakit menahun: Tidak ada. Evaluasi hambatan fungsional utama mencatat dimensi mengurus diri mandiri berstatus Tidak mengalami kesulitan, serta mobilisasi berjalan terpantau berstatus Tidak mengalami kesulitan.\"\n"
    "program_tidak_sesuai,Asistensi Sosial Penyandang Disabilitas (ASPD),TIDAK_ELIGIBLE,\"Alasan: Warga tidak memenuhi syarat program ASPD. Berdasarkan rekaman indikator fungsional, dimensi mengurus diri terpantau 'Tidak mengalami kesulitan' dan berjalan terpantau 'Tidak mengalami kesulitan', sehingga tidak masuk dalam kategori disabilitas berat yang membutuhkan asistensi sosial berkelanjutan.\"\n"
    "rekomendasi,PKH Plus (Lanjut Usia 70+),ELIGIBLE,\"Rank: 1 | Dasar Hukum: Juknis PKH Plus 2026 | Alasan: Warga atas nama ***ALI dinyatakan LAYAK menerima program PKH Plus karena secara kronologis telah berusia 77 tahun yang memenuhi syarat minimum juknis (70 tahun ke atas). Ditinjau dari aspek ekonomi, posisi rumah tangga berada pada desil 2 dengan status keberadaan Ditemukan / Aktif, yang merupakan klaster prioritas utama jaminan sosial Provinsi Jawa Timur.\"\n"
    "rekomendasi_teknis_bansos,Rencana_Aksi,-,\"Rekomendasi prioritas pemanfaatan ditujukan untuk pemenuhan kebutuhan dasar pokok, nutrisi gizi, serta layanan kesehatan utama penerima manfaat program PKH Plus Jatim. Otoritas penyaluran dan pemantauan lapangan berada di bawah koordinasi teknis Bank Jatim bersama Pendamping PKH dengan pendampingan melekat secara berkala oleh Pendamping Sosial PKH untuk menjamin ketepatan penggunaan dana. Alokasi dana bantuan sebesar Rp 500.000 per tahap (dengan total alokasi Rp 2.000.000 per tahun untuk 4 kali pencairan) wajib diprioritaskan untuk belanja pangan sehat serta biaya kontrol medis rutin, didukung langkah mitigasi berupa edukasi intensif kepala keluarga guna mencegah penyelewengan dana pada pos non-prioritas. Evaluasi dilakukan via uji petik lapangan dan pemutakhiran data verifikasi berkala pada setiap pergantian tahap pencairan guna memastikan hasil audit klasterisasi tetap valid.\""
)


# ============================================================
# PROMPT TEMPLATES (TOON format)
# ============================================================

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    f"{FEW_SHOT_USER_EXAMPLE}<|im_end|>\n"
    "<|im_start|>assistant\n"
    f"{FEW_SHOT_ASSISTANT_EXAMPLE}<|im_end|>\n"
    "<|im_start|>user\n"
    "=== PROFIL WARGA ===\n"
    "{query}\n"
    "=== AKHIR PROFIL WARGA ===\n\n"
    "=== KONTEKS DOKUMEN KEBIJAKAN DARI RETRIEVAL ===\n"
    "<dokumen_juknis>\n"
    "{context}\n"
    "</dokumen_juknis>\n"
    "=== AKHIR KONTEKS DOKUMEN ===\n\n"
    "INSTRUKSI EKSEKUSI:\n"
    "1. Lakukan audit kelayakan secara objektif dengan mencocokkan kriteria pada Profil Warga terhadap aturan di Konteks Dokumen.\n"
    "2. Hasilkan output TEPAT dalam format Toon dengan empat kategori wajib: 'ringkasan_profil', 'rekomendasi', 'rekomendasi_teknis_bansos', dan 'program_tidak_sesuai'.\n"
    "3. Pada baris 'rekomendasi', Anda wajib memuat informasi: rank, dasar hukum, dan alasan kelayakan di dalam kolom Detail/Alasan.\n"
    "4. Pada baris 'program_tidak_sesuai', Anda wajib memuat informasi: alasan ketidaksesuaian di dalam kolom Detail/Alasan.\n"
    "5. Respons hanya berupa teks format Toon valid. Jangan pernah memakai tag markdown pembungkus (seperti ```toon ...), heading teks tambahan di luar struktur, atau placeholder.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

POLICY_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    f"{FEW_SHOT_USER_EXAMPLE}<|im_end|>\n"
    "<|im_start|>assistant\n"
    f"{FEW_SHOT_ASSISTANT_EXAMPLE}<|im_end|>\n"
    "<|im_start|>user\n"
    "=== PROFIL WARGA ===\n"
    "{scoring_result}\n"
    "=== AKHIR PROFIL WARGA ===\n\n"
    "=== KONTEKS DOKUMEN KEBIJAKAN DARI RETRIEVAL ===\n"
    "<dokumen_juknis>\n"
    "{context}\n"
    "</dokumen_juknis>\n"
    "=== AKHIR KONTEKS DOKUMEN ===\n\n"
    "INSTRUKSI EKSEKUSI:\n"
    "1. Lakukan audit kelayakan secara objektif dengan mencocokkan kriteria pada Profil Warga terhadap aturan di Konteks Dokumen.\n"
    "2. Hasilkan output TEPAT dalam format Toon dengan empat kategori wajib: 'ringkasan_profil', 'rekomendasi', 'rekomendasi_teknis_bansos', dan 'program_tidak_sesuai'.\n"
    "3. Pada baris 'rekomendasi', Anda wajib memuat informasi: rank, dasar hukum, dan alasan kelayakan di dalam kolom Detail/Alasan.\n"
    "4. Pada baris 'program_tidak_sesuai', Anda wajib memuat informasi: alasan ketidaksesuaian di dalam kolom Detail/Alasan.\n"
    "5. Respons hanya berupa teks format Toon valid. Jangan pernah memakai tag markdown pembungkus (seperti ```toon ...), heading teks tambahan di luar struktur, atau placeholder.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

SYSTEM_PROMPT = (
    "[TASK_KLASIFIKASI_BANTUAN]\n"
    "Anda adalah AI Auditor resmi Dinas Sosial Provinsi Jawa Timur yang bertugas melakukan verifikasi dan validasi kelayakan penerima manfaat dua program bantuan sosial.\n\n"
    "Tugas Anda: Berdasarkan PROFIL WARGA dan KONTEKS PROGRAM BANTUAN (RETRIEVAL) yang disediakan, evaluasi kelayakan warga HANYA untuk 2 program utama berikut:\n"
    "1. Asistensi Sosial Penyandang Disabilitas (ASPD)\n"
    "2. PKH Plus (Lanjut Usia 70+)\n\n"
    "=== INSTRUKSI PENTING ===\n"
    "1. Evaluasi hanya 2 program utama di atas secara individual.\n"
    "2. Tentukan status: \"ELIGIBLE\" atau \"TIDAK_ELIGIBLE\".\n"
    "3. Ranking dari yang paling cocok ke yang paling tidak cocok.\n"
    "4. Berikan reasoning yang jelas dan WAJIB mengutip sumber dokumen resmi juknis.\n"
    "5. JANGAN merekomendasikan program bantuan di luar 2 program utama tersebut.\n"
    "6. DILARANG KERAS menyebut Program Sembako, PKH reguler, BPNT, PBI Jaminan Kesehatan, Rutilahu, PIP, Jamkesda, atau bantuan tambahan lainnya.\n\n"
    "=== FORMAT OUTPUT ===\n"
    "Anda WAJIB merespons HANYA dengan TEPAT menggunakan format Toon berikut. Tidak boleh ada markdown (seperti ```toon atau ```) dan dilarang keras menambahkan teks pembuka/penutup.\n\n"
    "Hasil[X]{Kategori,Nilai/Program,Status,Detail/Alasan}:\n"
    "ringkasan_profil,Profil_Warga,-,\"[string konkret berisi umur, desil, status DTSEN, disabilitas/usia lansia, dan kondisi kunci warga]\"\n"
    "rekomendasi,<Nama Program>,<Status>,\"Rank: <Angka> | Dasar Hukum: <Sumber> | Alasan: <Reasoning kelayakan>\"\n"
    "rekomendasi_teknis_bansos,Rencana_Aksi,-,\"[string narasi tunggal (paragraf utuh tanpa objek/poin berlapis) yang menjabarkan rencana aksi operasional, prioritas pemanfaatan dana, mekanisme pendampingan, pengelola bantuan, serta monitoring evaluasi warga di lapangan. Jika warga tidak berhak menerima program bantuan apa pun, maka nilai ini WAJIB disetel null]\"\n"
    "program_tidak_sesuai,<Nama Program>,<Status>,\"Alasan: <Reasoning ketidaksesuaian yang merujuk pada kondisi riil warga dan kriteria dokumen>\"\n\n"
    "=== LARANGAN KERAS ===\n"
    "- Jangan menyalin placeholder seperti \"Nama Program\", \"Rp X.XXX.XXX\", \"dst\", \"rangkuman singkat\", atau \"Penjelasan mengapa\".\n"
    "- Jangan mengosongkan alasan. Semua alasan harus merujuk kondisi riil warga dan kriteria dokumen.\n"
    "- nama_program harus ditulis persis salah satu dari 2 program utama yang disebut di atas."
)


# ============================================================
# KONFIGURASI PDF EXTRACTION (Stage 00)
# ============================================================
PDF_INPUT_DIR = str(BASE_DIR / "pdf_input")
SURYA_BATCH_SIZE = 2  # Sangat rendah untuk menghindari OOM 
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