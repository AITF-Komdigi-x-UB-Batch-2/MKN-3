# SIRA RAG MKN3

Repository ini difokuskan ke jalur RAG biasa:

Profil warga -> retrieval Qdrant -> generation API RunPod.

Output rekomendasi dibatasi ke 6 program utama. RAG tidak lagi membuat bagian
rekomendasi bantuan tambahan dari regulasi.

Agentic RAG belum dipakai di jalur utama. File terkait Agentic dan retriever lama
disimpan di `_archive/`.

## Jalur Utama

```text
config.py                  # konfigurasi path, model, prompt, Qdrant, RunPod
00_pdf_to_jsonl.py         # ekstraksi PDF regulasi dari pdf_input/
00_juknis_to_jsonl.py      # ekstraksi 6 Juknis utama dari pdf_juknis/
clean_jsonl.py             # normalisasi Juknis hasil ekstraksi
01_quality_check.py        # cek kualitas JSONL
02_EDA.py                  # EDA JSONL
03_normalize_jsonl.py      # normalisasi teks regulasi
04_embed_and_ingest_v2.py  # embedding dan ingest ke Qdrant
retrieval.py               # semantic search
generation.py              # helper prompt/context RAG
webservice.py              # FastAPI untuk RAG
evaluation/                # evaluasi RAG
_archive/                  # file lama/Agentic yang tidak aktif
```

## Model

- RAG biasa memakai API model Tim 1/RunPod via `TIM1_GENERATION_API_URL`.
- Embedding default: `BAAI/bge-m3`.

## Quick Start

1. Jalankan Qdrant:

```bash
docker compose up -d
```

2. Cek konfigurasi:

```bash
python config.py
```

3. Ekstrak regulasi dari `pdf_input/`:

```bash
python 00_pdf_to_jsonl.py
```

4. Ekstrak dan bersihkan Juknis utama dari `pdf_juknis/`:

```bash
python 00_juknis_to_jsonl.py
python clean_jsonl.py
```

5. Ingest JSONL ke Qdrant:

```bash
python 04_embed_and_ingest_v2.py
```

6. Jalankan RAG biasa:

```bash
python generation.py
```

7. Opsional, jalankan API:

```bash
uvicorn webservice:app --host 0.0.0.0 --port 8000
```

Swagger UI tersedia di `http://localhost:8000/docs`.

## Evaluasi

Validasi ground truth tanpa memanggil model:

```bash
python evaluation/run_rag_eval.py --dry-run --limit 1
```

Jalankan evaluasi RAG:

```bash
python evaluation/run_rag_eval.py
```
