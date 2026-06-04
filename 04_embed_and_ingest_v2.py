# ============================================================
# 04_embed_and_ingest_v2.py — Stage D: Embedding & Vector DB Ingestion (v2)
# Social Welfare Policy Recommender System (Tim 3)
#
# Input : file JSONL di chunked_data/
# Output: Qdrant vector database (remote via HTTP)
#
# Perubahan dari v1:
#   - UUID5 deterministik berbasis kolom Isi (text)
#
# Perubahan v2 → v3 (metadata baru dari clean_jsonl v2):
#   - Payload index untuk field nama_bansos (keyword) dan
#     tipe_konten (keyword array) agar filter Qdrant cepat
#   - augment_chunk_text() diperkaya dengan nama_bansos dari metadata
#     supaya embedding punya sinyal program bansos yang lebih kuat
#
# Backend embedding: fastembed (ONNX, CPU) via QdrantClient
#   - Tidak perlu torch / CUDA / langchain_huggingface
#   - Model di-cache di Temp/fastembed_cache (ONNX format)
#   - Named vector: key = nama model (dari get_fastembed_vector_params)
# ============================================================

import os
import re
import json
import uuid
import time
import sys
import logging
from pathlib import Path
from tqdm.auto import tqdm

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    PayloadSchemaType,
)

from config import (
    CHUNKED_DIR, QDRANT_URL, QDRANT_COLLECTION,
    EMBED_MODEL_NAME, EMBED_DIMENSIONS,
    EMBED_BATCH_SIZE, UPLOAD_BATCH_SIZE,
    ensure_dirs,
)


# ============================================================
# LOGGING
# ============================================================

def configure_utf8_stdio() -> None:
    """
    Pastikan print/log dengan emoji aman di Windows console non-UTF-8.
    Ini menggantikan kebutuhan menjalankan Python dengan -X utf8.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_stdio()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# FUNGSI UTILITAS
# ============================================================

def load_jsonl(filepath: str) -> list[dict]:
    """
    Baca file .jsonl → daftar dict {text, metadata}.
    Kompatibel dengan output clean_jsonl.py.
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" not in obj or "metadata" not in obj:
                    logger.warning(
                        "Baris %d di %s: key 'text'/'metadata' tidak ditemukan, dilewati.",
                        line_num, Path(filepath).name,
                    )
                    continue
                records.append(obj)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Baris %d di %s: JSON error (%s), dilewati.",
                    line_num, Path(filepath).name, e,
                )
    return records


def text_to_uuid(text: str) -> str:
    """
    Konversi teks → UUID5 deterministik.
    Memungkinkan idempotent re-run (upsert, bukan duplikasi).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def load_chunks_from_csv_ingestion(chunks: list[dict]) -> list[dict]:
    """
    Terima output langsung dari csv_ingestion (list of dicts).
    Validasi format: setiap dict harus memiliki 'text' dan 'metadata'.
    """
    valid = []
    for i, chunk in enumerate(chunks):
        if "text" not in chunk or "metadata" not in chunk:
            logger.warning("Chunk %d: format tidak valid, dilewati.", i)
            continue
        valid.append(chunk)
    return valid


# ============================================================
# SEMANTIC AUGMENTATION
# ============================================================

# Pola konten → tag semantik (ditambahkan ke TEKS sebelum embed, bukan metadata)
SEMANTIC_AUGMENT_RULES = [
    (
        re.compile(r'Rp\.?\s*[\d.,]+', re.IGNORECASE),
        "[nominal bantuan sosial besaran dana]"
    ),
    (
        re.compile(
            r'(usia produktif|keluarga rentan|miskin|tidak mampu|rentan)',
            re.IGNORECASE
        ),
        "[kriteria sasaran penerima]"
    ),
    (
        re.compile(
            r'(KTP|KK|kartu keluarga|NIK|surat domisili)',
            re.IGNORECASE
        ),
        "[syarat dokumen persyaratan]"
    ),
    (
        re.compile(
            r'(rekening|transfer|bank|pencairan|penyaluran)',
            re.IGNORECASE
        ),
        "[mekanisme pencairan penyaluran]"
    ),
    (
        re.compile(
            r'(DTSEN|DTKS|data terpadu|desil)',
            re.IGNORECASE
        ),
        "[data kemiskinan desil eligibilitas]"
    ),
]

# Mapping nama_bansos → tag semantik program
# Dipakai untuk inject sinyal program ke teks embedding
# supaya cosine similarity lebih kuat saat query menyebut nama program
BANSOS_AUGMENT_TAGS = {
    "ASPD":              "[asistensi sosial penyandang disabilitas ASPD]",
    "Kemiskinan Ekstrem":"[penanganan kemiskinan ekstrem miskin ekstrem desil 1]",
    "PKH Plus":          "[PKH Plus lanjut usia lansia 70 tahun]",
    "KIP KPM Jawara":    "[KIP KPM JAWARA kewirausahaan keluarga penerima manfaat]",
    "KIP PPKS Jawara":   "[KIP PPKS JAWARA pemerlu pelayanan kesejahteraan sosial]",
    "KIP Putri Jawara":  "[KIP Putri JAWARA perempuan tangguh wirausaha]",
}


def augment_chunk_text(text: str, metadata: dict | None = None) -> str:
    """
    Tambahkan tag semantik ke teks sebelum embedding.

    Dua lapisan augmentasi:
    1. Rule-based: deteksi pola Rp, KTP, rekening, dll → tag generik
    2. Metadata-based: pakai nama_bansos dari metadata → tag program spesifik

    Catatan: augmentasi HANYA untuk embedding, tidak disimpan ke payload.
    Payload Qdrant tetap menyimpan teks asli (texts_original).
    """
    tags_to_add = set()

    # Layer 1: Rule-based (sama seperti sebelumnya)
    for pattern, tag in SEMANTIC_AUGMENT_RULES:
        if pattern.search(text):
            tags_to_add.add(tag)

    # Layer 2: nama_bansos dari metadata (BARU)
    # Inject tag program yang spesifik sehingga embedding chunk ASPD
    # lebih dekat ke query yang menyebut "ASPD" atau "disabilitas"
    if metadata:
        nama_bansos = metadata.get("nama_bansos", "")
        if nama_bansos and nama_bansos in BANSOS_AUGMENT_TAGS:
            tags_to_add.add(BANSOS_AUGMENT_TAGS[nama_bansos])

    if tags_to_add:
        return text + "\n" + " ".join(sorted(tags_to_add))

    return text


# ============================================================
# INISIALISASI MODEL & DATABASE
# ============================================================

def init_embedding_model() -> TextEmbedding:
    """
    Load model fastembed (ONNX, CPU).
    Model di-cache di Temp/fastembed_cache — tidak perlu torch/CUDA.
    """
    logger.info("📦 Memuat model fastembed: %s ...", EMBED_MODEL_NAME)
    model = TextEmbedding(
        model_name=EMBED_MODEL_NAME,
        threads=None,        # pakai semua CPU core
    )
    logger.info("✅ Model fastembed siap (ONNX, CPU).")
    return model


def init_qdrant() -> QdrantClient:
    """
    Inisialisasi Qdrant client via HTTP.
    Pakai fastembed (named vector) — kompatibel dengan ingestion.py.
    Collection dibuat dengan get_fastembed_vector_params() agar
    vector name otomatis sesuai model name.
    """
    logger.info("📦 Menghubungkan ke Qdrant server (%s) ...", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL)

    # Set fastembed model agar client tahu vector name & dimensi
    client.set_model(EMBED_MODEL_NAME)

    if client.collection_exists(QDRANT_COLLECTION):
        count = client.get_collection(QDRANT_COLLECTION).points_count
        logger.info(
            "✅ Collection '%s' sudah ada (%d points).",
            QDRANT_COLLECTION, count,
        )
    else:
        # Buat collection dengan named vector (fastembed format)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=client.get_fastembed_vector_params(),
        )
        logger.info(
            "✅ Collection '%s' dibuat (fastembed named vector).",
            QDRANT_COLLECTION,
        )

    # ── Payload Index ──────────────────────────────────────
    # Index field yang sering dipakai untuk filter di generation.py
    # agar filter tidak melakukan full scan (O(n) → O(log n))
    #
    # nama_bansos : Keyword  → filter exact match, e.g. "ASPD"
    # tipe_konten : Keyword  → filter MatchAny pada list, e.g. ["kriteria_penerima"]
    # sumber      : Keyword  → filter lama (tetap dipertahankan)
    #
    # create_payload_index idempotent — aman dijalankan ulang.
    _ensure_payload_indexes(client)

    return client


def _ensure_payload_indexes(client: QdrantClient) -> None:
    """
    Buat payload index untuk field filter yang sering dipakai.
    Idempotent: aman dipanggil meski index sudah ada.
    """
    fields_to_index = [
        ("nama_bansos", PayloadSchemaType.KEYWORD),
        ("tipe_konten", PayloadSchemaType.KEYWORD),          # array of keyword
        ("tipe_konten_primer", PayloadSchemaType.KEYWORD),   # label utama dari clean_jsonl
        ("retrieval_priority", PayloadSchemaType.KEYWORD),   # normal / low
        ("quality_flags", PayloadSchemaType.KEYWORD),        # array, hanya ada di chunk low priority
        ("sumber", PayloadSchemaType.KEYWORD),               # filter lama
    ]

    for field_name, schema_type in fields_to_index:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.info("  📇 Payload index: '%s' (%s) ✅", field_name, schema_type)
        except Exception as e:
            # Index sudah ada → tidak perlu diulang, bukan error fatal
            logger.debug("  Payload index '%s' sudah ada atau gagal: %s", field_name, e)


# ============================================================
# BATCH INGESTION
# ============================================================

def ingest_chunks(
    chunks: list[dict],
    client: QdrantClient,
    embeddings: TextEmbedding,
    source_name: str = "csv_data",
) -> int:
    """
    Proses daftar chunk dicts:
      1. Augment teks untuk embedding (sinyal semantik lebih kuat)
      2. Embed teks augmented via fastembed (ONNX, CPU)
      3. Upload ke Qdrant dalam batch UPLOAD_BATCH_SIZE
      4. ID deterministik berbasis UUID5 dari teks asli
      5. Payload menyimpan teks ASLI + semua metadata

    Catatan: augmented text hanya untuk embedding, tidak disimpan.
    Returns: jumlah points yang berhasil di-upload.
    """
    if not chunks:
        logger.warning("⚠️ %s: Tidak ada chunk valid.", source_name)
        return 0

    logger.info("📄 %s: %d chunks ditemukan.", source_name, len(chunks))

    # Teks asli → disimpan ke payload Qdrant
    texts_original = [c["text"] for c in chunks]

    # Teks augmented → hanya untuk embedding (lebih kaya sinyal semantik)
    texts_for_embed = [
        augment_chunk_text(c["text"], metadata=c.get("metadata"))
        for c in chunks
    ]

    metadatas = [c["metadata"] for c in chunks]

    # ── Embedding via fastembed (ONNX) ─────────────────────
    logger.info(
        "   🔢 Embedding %d teks dengan fastembed ...",
        len(texts_for_embed),
    )
    embed_start = time.time()

    # fastembed .embed() mengembalikan generator numpy array
    # tqdm wrap untuk progress bar
    raw_vecs = embeddings.embed(
        texts_for_embed,
        batch_size=EMBED_BATCH_SIZE if EMBED_BATCH_SIZE > 1 else 32,
    )
    vectors = [vec.tolist() for vec in tqdm(raw_vecs, total=len(texts_for_embed), desc="Embedding", unit="chunk")]

    embed_time = time.time() - embed_start
    logger.info("   ✅ Embedding selesai dalam %.1f detik.", embed_time)

    # Ambil nama vector dari fastembed params (e.g. "intfloat/multilingual-e5-large")
    vector_name = list(client.get_fastembed_vector_params().keys())[0]

    # ── Upload ke Qdrant dalam batch ──────────────────────
    total_uploaded = 0
    num_batches = (len(chunks) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE

    for batch_idx in range(num_batches):
        start = batch_idx * UPLOAD_BATCH_SIZE
        end   = min(start + UPLOAD_BATCH_SIZE, len(chunks))

        points = []
        for i in range(start, end):
            point_id = text_to_uuid(texts_original[i])
            points.append(
                PointStruct(
                    id=point_id,
                    # named vector — wajib untuk collection fastembed
                    vector={vector_name: vectors[i]},
                    payload={
                        "text": texts_original[i],  # teks asli, bukan augmented
                        **metadatas[i],              # semua metadata termasuk
                                                     # nama_bansos & tipe_konten
                    },
                )
            )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )
        total_uploaded += len(points)
        logger.info(
            "   📤 Batch %d/%d uploaded (%d points, total %d).",
            batch_idx + 1, num_batches, len(points), total_uploaded,
        )

    return total_uploaded


def ingest_file(
    filepath: str,
    client: QdrantClient,
    embeddings,
) -> int:
    """Proses satu file .jsonl (kompatibilitas mundur)."""
    filename = Path(filepath).name
    chunks = load_jsonl(filepath)
    return ingest_chunks(chunks, client, embeddings, source_name=filename)


REQUIRED_CLEAN_METADATA_FIELDS = (
    "nama_bansos",
    "tipe_konten",
    "tipe_konten_primer",
    "retrieval_priority",
)

MAIN_JUKNIS_SOURCES = {
    "Juklak ASPD Tahun 202620260225_12303533_01.pdf",
    "JUKNIS KEMISKINAN EKSTREM (13-1-2025)-1 (1) (2).pdf",
    "JUKNIS PKH PLUS 2026.pdf",
    "PETUNJUK TEKNIS KIP KPM JAWARA.pdf",
    "Petunjuk Teknis KIP PPKS Jawara 2026.pdf",
    "PETUNJUK TEKNIS KIP PUTRI JAWARA.pdf",
}


def requires_clean_metadata(metadata: dict) -> bool:
    """
    Metadata baru dari clean_jsonl.py wajib untuk 6 Juknis program utama.
    Dokumen regulasi pendukung boleh tidak punya nama_bansos/tipe_konten.
    """
    return (
        metadata.get("sumber") in MAIN_JUKNIS_SOURCES
        or bool(metadata.get("nama_bansos"))
    )


def validate_clean_metadata(chunks: list[dict]) -> bool:
    """
    Validasi metadata hasil clean_jsonl.py pada seluruh chunk Juknis utama.
    JSONL regulasi pendukung tetap boleh masuk walau tidak punya metadata
    nama_bansos/tipe_konten, karena memang bukan keluaran clean_jsonl.py.
    """
    missing_counts = {field: 0 for field in REQUIRED_CLEAN_METADATA_FIELDS}
    invalid_tipe_konten = 0
    invalid_priority = 0
    checked_program_chunks = 0
    support_chunks = 0

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if not requires_clean_metadata(metadata):
            support_chunks += 1
            continue

        checked_program_chunks += 1

        for field in REQUIRED_CLEAN_METADATA_FIELDS:
            if field not in metadata or metadata.get(field) in ("", None, []):
                missing_counts[field] += 1

        tipe_konten = metadata.get("tipe_konten")
        if not isinstance(tipe_konten, list) or not tipe_konten:
            invalid_tipe_konten += 1

        if metadata.get("retrieval_priority") not in {"normal", "low"}:
            invalid_priority += 1

    has_error = (
        any(missing_counts.values())
        or invalid_tipe_konten > 0
        or invalid_priority > 0
    )

    print("\n📋 Verifikasi metadata chunk:")
    print(f"   Juknis utama dicek : {checked_program_chunks} chunk")
    print(f"   Regulasi pendukung : {support_chunks} chunk (metadata baru opsional)")
    for field, count in missing_counts.items():
        status = "✅ lengkap" if count == 0 else f"⚠️ kurang di {count} chunk"
        print(f"   {field:20s}: {status}")
    print(
        "   tipe_konten format : "
        f"{'✅ list valid' if invalid_tipe_konten == 0 else f'⚠️ invalid di {invalid_tipe_konten} chunk'}"
    )
    print(
        "   retrieval_priority : "
        f"{'✅ valid' if invalid_priority == 0 else f'⚠️ invalid di {invalid_priority} chunk'}"
    )

    if has_error:
        print("\n❌ Metadata baru belum lengkap. Batalkan ingest.")
        print("   → Jalankan clean_jsonl.py untuk 6 Juknis program utama,")
        print("     lalu pastikan juknis_extracted_normalized.jsonl yang dipakai.\n")
        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dirs()
    start_all = time.time()

    print("=" * 65)
    print("🚀 STAGE D v2: Embedding & Vector DB Ingestion")
    print(f"   Input     : {CHUNKED_DIR}")
    print(f"   Output    : {QDRANT_URL}")
    print(f"   Collection: {QDRANT_COLLECTION}")
    print(f"   Model     : {EMBED_MODEL_NAME} ({EMBED_DIMENSIONS}d)")
    print(f"   Batch     : embed={EMBED_BATCH_SIZE}, upload={UPLOAD_BATCH_SIZE}")
    print("=" * 65)

    chunks = []

    if not os.path.isdir(CHUNKED_DIR):
        raise FileNotFoundError(
            f"❌ Direktori input tidak ditemukan: {CHUNKED_DIR}\n"
            f"   Jalankan csv_ingestion.py terlebih dahulu."
        )

    jsonl_files = sorted([
        f for f in os.listdir(CHUNKED_DIR)
        if f.lower().endswith(".jsonl")
    ])

    if not jsonl_files:
        logger.warning("⚠️ Tidak ada file .jsonl di folder input.")
        return

    # Filter: kalau ada versi _normalized, skip file aslinya
    normalized_suffix = "_normalized.jsonl"
    normalized_files  = [f for f in jsonl_files if f.endswith(normalized_suffix)]

    final_jsonl_files = []
    for f in jsonl_files:
        if not f.endswith(normalized_suffix):
            expected_norm = f.replace(".jsonl", normalized_suffix)
            if expected_norm in normalized_files:
                logger.info("⏭️ Melewati %s → ada versi %s", f, expected_norm)
                continue
        final_jsonl_files.append(f)

    for f in final_jsonl_files:
        filepath    = os.path.join(CHUNKED_DIR, f)
        file_chunks = load_jsonl(filepath)
        chunks.extend(file_chunks)

    logger.info(
        "✅ %d chunks dimuat dari %d file JSONL.",
        len(chunks), len(final_jsonl_files),
    )

    if not chunks:
        print("\n⚠️ Tidak ada chunk yang dapat diproses.")
        return

    # ── Verifikasi metadata baru ──────────────────────────
    if not validate_clean_metadata(chunks):
        return

    # ── Inisialisasi ──────────────────────────────────────
    embeddings = init_embedding_model()
    client     = init_qdrant()

    # ── Ingest ────────────────────────────────────────────
    total_points = ingest_chunks(
        chunks, client, embeddings, source_name="all_chunks"
    )

    elapsed = time.time() - start_all

    # ── Ringkasan ─────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("📊 STAGE D v2 SELESAI")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Total points : {total_points:,}")
    print(f"   Waktu total  : {elapsed:.1f} detik")

    # Verifikasi collection
    try:
        info  = client.get_collection(QDRANT_COLLECTION)
        vconf = info.config.params.vectors
        # fastembed pakai named vector (dict), bukan unnamed VectorParams
        if isinstance(vconf, dict):
            vec_name  = list(vconf.keys())[0]
            vec_size  = list(vconf.values())[0].size
            vec_dist  = list(vconf.values())[0].distance
        else:
            vec_name  = "(default)"
            vec_size  = vconf.size
            vec_dist  = vconf.distance
        print(f"\n📊 Verifikasi Collection '{QDRANT_COLLECTION}':")
        print(f"   Points count  : {info.points_count:,}")
        print(f"   Vector name   : {vec_name}")
        print(f"   Vectors size  : {vec_size}")
        print(f"   Distance      : {vec_dist}")
        print(f"   Status        : {info.status}")
        print(f"\n   Payload index  :")
        print(f"     - nama_bansos  (keyword) → filter program bansos")
        print(f"     - tipe_konten  (keyword) → filter tipe konten chunk")
        print(f"     - tipe_konten_primer (keyword) → filter label utama chunk")
        print(f"     - retrieval_priority (keyword) → filter chunk normal/low")
        print(f"     - quality_flags (keyword) → filter catatan kualitas chunk")
        print(f"     - sumber       (keyword) → filter nama file PDF")
    except Exception as e:
        logger.error("Gagal verifikasi collection: %s", e)

    print(f"\n✅ Pipeline Stage D v2 selesai! DB: {QDRANT_URL}")


if __name__ == "__main__":
    main()
