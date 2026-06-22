# 🌟 SIRA RAG (Sistem Intervensi & Kebijakan Program Sosial)
**Tim 3 Universitas Brawijaya (AITF 2026)**

SIRA adalah sebuah sistem berbasis **Retrieval-Augmented Generation (RAG)** yang dirancang untuk membantu Dinas Sosial (atau lembaga terkait) mengevaluasi kelayakan warga untuk menerima program bantuan sosial (bansos). Sistem ini mencocokkan **Profil Warga** dengan aturan resmi yang ada di dalam **Petunjuk Teknis (Juknis)**.

Saat ini, sistem difokuskan secara spesifik untuk merekomendasikan **2 program utama**:
1. **Asistensi Sosial Penyandang Disabilitas (ASPD)**
2. **PKH Plus (Lanjut Usia 70+)**

---

## 🏗️ Arsitektur Sistem

Pipeline sistem ini mengalir dari dokumen PDF mentah hingga menjadi REST API yang siap digunakan:

1. **Document Ingestion (Ekstraksi)**: Mengekstrak teks digital dan hasil OCR dari dokumen regulasi/Juknis berformat PDF menjadi format terstruktur (JSONL).
2. **Preprocessing & Embedding**: Normalisasi teks, pemotongan (*chunking*), dan proses *embedding* (vektorisasi) menggunakan model `intfloat/multilingual-e5-large` (via FastEmbed).
3. **Vector Database**: Data vektor dan metadata di-*ingest* secara lokal ke dalam **Qdrant** untuk memungkinkan pencarian konteks secara semantik (*Semantic Search*).
4. **Generation & Guardrails**: Menggunakan Large Language Model (LLM) eksternal via API (RunPod/Tim 1) dengan *Few-Shot Prompting* untuk menghasilkan rekomendasi yang terstruktur (*TOON format*). Terdapat pengamanan *(guardrails)* untuk mencegah AI merekomendasikan program di luar kewenangannya.
5. **REST API**: Mengekspos seluruh pipeline melalui **FastAPI**.

---

## 📂 Struktur Repositori Utama

```text
├── ⚙️ Konfigurasi Utama
│   └── config.py                  # Konfigurasi terpusat (Path, Qdrant, Model, Prompt)
├── 📥 Ingestion & Preprocessing
│   ├── 00_pdf_to_jsonl.py         # Ekstraksi PDF regulasi umum (dari pdf_input/)
│   ├── 00_juknis_to_jsonl.py      # Ekstraksi PDF Juknis spesifik (dari pdf_juknis/)
│   ├── clean_jsonl.py             # Normalisasi & pembersihan JSONL Juknis
│   ├── 01_quality_check.py        # Validasi kualitas data hasil ekstraksi
│   ├── 02_EDA.py                  # Analisis Data Eksploratif pada JSONL
│   ├── 03_normalize_jsonl.py      # Normalisasi akhir teks sebelum embedding
│   └── 04_embed_and_ingest_v2.py  # Embedding data teks dan penyimpanan ke Qdrant
├── 🧠 RAG & LLM Engine
│   ├── retrieval.py               # Modul Semantic Search ke Qdrant
│   ├── generation.py              # Helper untuk formatting Prompt, Konteks, dan Parsing LLM
│   ├── llm_client.py              # Klien pemanggil API LLM (RunPod)
│   └── guardrails.py              # Logika fallback & filter validasi program
├── 🌐 Web Service
│   └── webservice.py              # Aplikasi FastAPI (Endpoints: /recommend, /retrieve, /health)
├── 📊 Evaluasi
│   └── evaluation/                # Skrip evaluasi performa RAG
└── 🗄️ Arsip
    └── _archive/                  # Skrip RAG Agentic dan file lama yang tidak aktif
```

---

## 🚀 Panduan Memulai Cepat (Quick Start)

### 1. Prasyarat Sistem
- **Python 3.10+** (disarankan menggunakan *virtual environment*)
- **Docker & Docker Compose** (untuk menjalankan instance Qdrant secara lokal)

Instalasi dependensi Python:
```bash
pip install -r requirements.txt
# atau menggunakan uv:
# uv pip install -r requirements.txt
```

### 2. Konfigurasi Lingkungan
Buat file `.env` di direktori *root* (jika belum ada) dan sesuaikan nilainya (terutama *endpoint* dan kunci API LLM). Cek ketersediaan dan konsistensi konfigurasi dengan menjalankan:
```bash
python config.py
```

### 3. Menjalankan Database Qdrant
Jalankan kontainer Qdrant di latar belakang (port default: `6333`):
```bash
docker compose up -d
```

### 4. Menjalankan Pipeline Pemrosesan Data (ETL)
Jalankan langkah-langkah di bawah ini secara berurutan jika Anda baru pertama kali mengatur database atau memiliki dokumen Juknis PDF baru.

**A. Ekstraksi Dokumen**
```bash
# Ekstrak Regulasi Umum
python 00_pdf_to_jsonl.py

# Ekstrak Juknis Spesifik & Bersihkan Datanya
python 00_juknis_to_jsonl.py
python clean_jsonl.py
```

**B. Embedding & Ingestion ke Qdrant**
```bash
python 04_embed_and_ingest_v2.py
```

### 5. Menjalankan Uji Coba CLI
Untuk memastikan RAG bisa menarik dokumen dari Qdrant dan berinteraksi dengan LLM sebelum API dijalankan:
```bash
python generation.py
```

### 6. Menjalankan Web Service API (FastAPI)
Jika keseluruhan *pipeline* sudah sukses tertanam ke database, jalankan server API:
```bash
uvicorn webservice:app --host 0.0.0.0 --port 8002 --reload
```
Akses dokumentasi antarmuka interaktif (Swagger UI) di: **`http://localhost:8002/docs`**

---

## 💻 Penggunaan API (Gambaran Endpoint)

Sistem akan berjalan di `http://localhost:8002`. Berikut gambaran *endpoint* utamanya:

- **`POST /recommend`**: Mengirim profil warga (dalam format teks/deskripsi) dan mendapatkan balasan berupa *ranking* kesesuaian program (ASPD/PKH Plus), status kelayakan, alasan, dan rencana aksi teknis bansos berformat JSON terstruktur.
- **`POST /retrieve`**: Melakukan *Semantic Search* terhadap Juknis tanpa memanggil model LLM (berguna untuk *debugging* atau inspeksi kembalian *chunk* Qdrant).
- **`GET /health`**: Mengecek status ketersediaan *Service*, koneksi Qdrant, dan parameter Model.
- **`GET /programs`**: Melihat daftar program sosial yang didukung di dalam sistem saat ini.

---

## 📈 Evaluasi Performa RAG

Sistem ini memiliki modul *script* pengujian untuk mengevaluasi seberapa akurat sistem mencocokkan kondisi profil warga dengan *Ground Truth* di Juknis.

**A. Mode Dry-Run** (Validasi format data uji tanpa memanggil API LLM; berguna untuk hemat *cost/rate-limit*):
```bash
python evaluation/run_rag_eval.py --dry-run --limit 1
```

**B. Mode Evaluasi Penuh:**
```bash
python evaluation/run_rag_eval.py
```
