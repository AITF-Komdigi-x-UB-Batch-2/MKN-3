# ============================================================
# rebuild_juknis_points.py
# Hapus HANYA points juknis lama dari Qdrant, lalu embed ulang
# dari juknis_extracted_normalized.jsonl.
#
# Data regulasi (Permensos, Perpres, UU, Kepmensos) TIDAK disentuh.
#
# Kenapa tidak cukup upsert saja?
#   UUID5 dibuat dari teks chunk. Teks lama (tanpa prefix) dan teks
#   baru (dengan prefix [Sumber:...]) menghasilkan UUID berbeda.
#   Upsert biasa MENAMBAH point baru, tidak mengganti yang lama.
#   Akibatnya collection punya versi duplikat: dengan prefix & tanpa.
#
# Cara pakai:
#   python rebuild_juknis_points.py
# ============================================================

import os
import json
import uuid
import time
import logging
from pathlib import Path

# Set HF cache sebelum import apapun
HF_PATH = r"D:\AITF-KEMISKINAN\KEMISKINAN\MKN3_CSV\model_cache\huggingface"
os.environ["HF_HOME"] = HF_PATH
os.environ["HF_HUB_CACHE"] = os.path.join(HF_PATH, "hub")

from tqdm.auto import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)

from config import (
    CHUNKED_DIR, QDRANT_URL, QDRANT_COLLECTION,
    EMBED_MODEL_NAME, EMBED_DIMENSIONS,
    EMBED_BATCH_SIZE, UPLOAD_BATCH_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NORMALIZED_FILE = os.path.join(CHUNKED_DIR, "juknis_extracted_normalized.jsonl")
JUKNIS_KATEGORI = "Petunjuk Teknis (Juknis)"  # nilai metadata.kategori di juknis


# ============================================================
# STEP 1: Hapus points juknis lama
# ============================================================

def delete_old_juknis_points(client: QdrantClient) -> int:
    """
    Hapus semua points dengan metadata.kategori == 'Petunjuk Teknis (Juknis)'.
    Points regulasi (Regulasi, Aturan Hukum, dll) tidak tersentuh.
    """
    logger.info("🗑️  Menghapus points juknis lama dari collection '%s'...", QDRANT_COLLECTION)

    # Qdrant delete by filter
    result = client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="kategori",
                    match=MatchValue(value=JUKNIS_KATEGORI),
                )
            ]
        ),
    )

    # Hitung sisa points
    info = client.get_collection(QDRANT_COLLECTION)
    logger.info("✅ Delete selesai. Sisa points: %d", info.points_count)
    return info.points_count


# ============================================================
# STEP 2: Load normalized JSONL
# ============================================================

def load_normalized_jsonl() -> list[dict]:
    if not os.path.exists(NORMALIZED_FILE):
        raise FileNotFoundError(
            f"❌ File tidak ditemukan: {NORMALIZED_FILE}\n"
            f"   Jalankan clean_jsonl.py terlebih dahulu."
        )

    records = []
    with open(NORMALIZED_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj and "metadata" in obj:
                    records.append(obj)
            except json.JSONDecodeError as e:
                logger.warning("Baris %d: JSON error (%s), dilewati.", line_num, e)

    logger.info("📂 Loaded %d chunks dari %s", len(records), Path(NORMALIZED_FILE).name)
    return records


# ============================================================
# STEP 3: Embed & Upload
# ============================================================

def init_embedding_model():
    import torch
    import platform

    system = platform.system()
    if system == "Darwin" and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info("🖥️  Embedding device: %s", device.upper())

    if device in ("cuda", "mps"):
        model_kwargs = {
            "device": device,
            "model_kwargs": {"torch_dtype": torch.bfloat16},
        }
    else:
        model_kwargs = {"device": device}

    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info("📦 Memuat HuggingFace: %s ...", EMBED_MODEL_NAME)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": EMBED_BATCH_SIZE, "normalize_embeddings": True},
    )
    logger.info("✅ HuggingFace embedding siap (device=%s).", device)
    return embeddings


def text_to_uuid(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def embed_and_upload(chunks: list[dict], client: QdrantClient, embeddings) -> int:
    if not chunks:
        logger.warning("⚠️ Tidak ada chunk untuk di-embed.")
        return 0

    texts     = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info("🔢 Embedding %d chunks (batch_size=%d)...", len(texts), EMBED_BATCH_SIZE)
    t0 = time.time()

    if EMBED_BATCH_SIZE <= 1:
        vectors = []
        for text in tqdm(texts, desc="Embedding", unit="chunk"):
            vec = embeddings.embed_documents([text])
            vectors.append(vec[0])
    else:
        vectors = embeddings.embed_documents(texts)

    logger.info("✅ Embedding selesai: %.1f detik", time.time() - t0)

    # Upload dalam batch
    total = 0
    num_batches = (len(chunks) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE

    for b in range(num_batches):
        start = b * UPLOAD_BATCH_SIZE
        end   = min(start + UPLOAD_BATCH_SIZE, len(chunks))

        points = [
            PointStruct(
                id=text_to_uuid(texts[i]),
                vector=vectors[i],
                payload={"text": texts[i], **metadatas[i]},
            )
            for i in range(start, end)
        ]

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        total += len(points)
        logger.info("   📤 Batch %d/%d — %d points (total %d)", b+1, num_batches, len(points), total)

    return total


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()

    print("=" * 65)
    print("🔄 REBUILD JUKNIS POINTS")
    print(f"   Collection : {QDRANT_COLLECTION} @ {QDRANT_URL}")
    print(f"   Input      : {NORMALIZED_FILE}")
    print(f"   Embed Model: {EMBED_MODEL_NAME} ({EMBED_DIMENSIONS}d)")
    print("=" * 65)

    client = QdrantClient(url=QDRANT_URL)

    # Cek collection ada
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        logger.info("Collection belum ada, membuat baru...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIMENSIONS, distance=Distance.COSINE),
        )

    # Info sebelum
    info_before = client.get_collection(QDRANT_COLLECTION)
    print(f"\n📊 Sebelum: {info_before.points_count} points di collection")

    # Step 1: Hapus juknis lama
    remaining = delete_old_juknis_points(client)
    print(f"   → Setelah delete juknis lama: {remaining} points (regulasi tetap aman)")

    # Step 2: Load normalized
    chunks = load_normalized_jsonl()

    # Step 3: Embed & upload
    embeddings = init_embedding_model()
    uploaded   = embed_and_upload(chunks, client, embeddings)

    # Info sesudah
    info_after = client.get_collection(QDRANT_COLLECTION)
    elapsed    = time.time() - t_start

    print(f"\n{'=' * 65}")
    print("✅ REBUILD SELESAI")
    print(f"{'=' * 65}")
    print(f"   Points sebelum : {info_before.points_count}")
    print(f"   Juknis dihapus : {info_before.points_count - remaining}")
    print(f"   Juknis baru    : {uploaded}")
    print(f"   Points sesudah : {info_after.points_count}")
    print(f"   Waktu total    : {elapsed:.1f} detik")
    print(f"{'=' * 65}")
    print(f"\n👉 Selanjutnya: python retrieval.py")


if __name__ == "__main__":
    main()
