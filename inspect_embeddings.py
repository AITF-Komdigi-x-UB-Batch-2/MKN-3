# ============================================================
# inspect_embeddings.py — Inspeksi Embedding di Qdrant Server
# ============================================================
# Cara pakai:
#   python inspect_embeddings.py              → ringkasan collection
#   python inspect_embeddings.py --vectors    → tampilkan vektor (5 point pertama)
#   python inspect_embeddings.py --search "pertanyaan anda"  → cari via embedding
# ============================================================

import argparse
import json
import numpy as np

from qdrant_client import QdrantClient, models

from config import QDRANT_URL, QDRANT_COLLECTION, EMBED_DIMENSIONS, EMBED_MODEL_NAME


def get_client() -> QdrantClient:
    print(f"🔗 Menghubungkan ke Qdrant: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)
    return client


# ── 1. Ringkasan Collection ────────────────────────────────────

def show_collection_info(client: QdrantClient):
    """Tampilkan info collection: jumlah points, dimensi, dll."""
    print(f"\n{'=' * 60}")
    print("📊 INFO COLLECTION")
    print(f"{'=' * 60}")

    try:
        collections = client.get_collections().collections
        print(f"\n🗂  Collections yang ada ({len(collections)}):")
        for c in collections:
            print(f"   - {c.name}")
    except Exception as e:
        print(f"❌ Gagal list collections: {e}")
        return

    try:
        info = client.get_collection(QDRANT_COLLECTION)
        print(f"\n📋 Detail: '{QDRANT_COLLECTION}'")
        print(f"   Points count  : {info.points_count:,}")
        print(f"   Vectors size  : {info.config.params.vectors.size}")
        print(f"   Distance      : {info.config.params.vectors.distance}")
        print(f"   Status        : {info.status}")
    except Exception as e:
        print(f"❌ Collection '{QDRANT_COLLECTION}' tidak ditemukan: {e}")


# ── 2. Tampilkan Sample Vectors ────────────────────────────────

def show_sample_vectors(client: QdrantClient, n: int = 5):
    """Ambil n point pertama dan tampilkan vektor + metadata-nya."""
    print(f"\n{'=' * 60}")
    print(f"🔢 SAMPLE EMBEDDING VECTORS (n={n})")
    print(f"{'=' * 60}")

    try:
        results, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=n,
            with_vectors=True,      # ← ini yang menampilkan vektor embedding
            with_payload=True,
        )
    except Exception as e:
        print(f"❌ Gagal mengambil data: {e}")
        return

    if not results:
        print("⚠️  Collection kosong — jalankan ingest terlebih dahulu.")
        return

    for i, point in enumerate(results, 1):
        vec = np.array(point.vector)
        payload = point.payload or {}
        text = payload.get("text", "")[:100]
        sumber = payload.get("Sumber", payload.get("sumber", "-"))

        print(f"\n── Point #{i} ─────────────────────────────────────────")
        print(f"   ID       : {point.id}")
        print(f"   Sumber   : {sumber}")
        print(f"   Teks     : {text}{'...' if len(payload.get('text','')) > 100 else ''}")
        print(f"   Dimensi  : {len(vec)}")
        print(f"   Min      : {vec.min():.6f}")
        print(f"   Max      : {vec.max():.6f}")
        print(f"   Mean     : {vec.mean():.6f}")
        print(f"   Std      : {vec.std():.6f}")
        print(f"   Norm (L2): {np.linalg.norm(vec):.6f}")
        # Tampilkan 10 nilai pertama vektor
        preview = ", ".join(f"{v:.4f}" for v in vec[:10])
        print(f"   Vector[:10]: [{preview}, ...]")


# ── 3. Semantic Search via Embedding ──────────────────────────

def search_by_query(client: QdrantClient, query: str, top_k: int = 3):
    """Embed query dengan FastEmbed lalu cari di Qdrant."""
    print(f"\n{'=' * 60}")
    print(f"🔍 SEMANTIC SEARCH")
    print(f"{'=' * 60}")
    print(f"   Query  : \"{query}\"")
    print(f"   Top-K  : {top_k}")

    try:
        client.set_model(EMBED_MODEL_NAME)
        vector_name = list(client.get_fastembed_vector_params().keys())[0]
        print(f"   Model  : {EMBED_MODEL_NAME} (FastEmbed)\n")
    except Exception as e:
        print(f"❌ Gagal menyiapkan embedding model: {e}")
        return

    try:
        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=models.Document(text=query, model=EMBED_MODEL_NAME),
            using=vector_name,
            limit=top_k,
            with_payload=True,
        ).points
    except Exception as e:
        print(f"❌ Gagal search: {e}")
        return

    if not hits:
        print("\n⚠️  Tidak ada hasil.")
        return

    for i, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        text = payload.get("text", "")
        sumber = payload.get("Sumber", payload.get("sumber", "-"))
        print(f"\n   📄 Hasil #{i} — Score: {hit.score:.4f}")
        print(f"      Sumber : {sumber}")
        print(f"      Teks   : {text[:200]}{'...' if len(text) > 200 else ''}")


# ── 4. Export Vectors ke JSON ──────────────────────────────────

def export_vectors(client: QdrantClient, output_path: str = "embeddings_export.json", limit: int = 50):
    """Export sejumlah point (id + vector + payload) ke file JSON."""
    print(f"\n{'=' * 60}")
    print(f"💾 EXPORT VECTORS → {output_path}")
    print(f"{'=' * 60}")

    exported = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=min(100, limit - len(exported)),
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for point in results:
            exported.append({
                "id": str(point.id),
                "vector": point.vector,
                "payload": point.payload,
            })

        if next_offset is None or len(exported) >= limit:
            break
        offset = next_offset

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)

    print(f"   ✅ {len(exported)} points diekspor ke: {output_path}")


# ── MAIN ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspeksi embedding Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python inspect_embeddings.py                        # info collection saja
  python inspect_embeddings.py --vectors              # lihat 5 sample vektor
  python inspect_embeddings.py --vectors --n 10       # lihat 10 sample vektor
  python inspect_embeddings.py --search "PKH 2025"   # semantic search
  python inspect_embeddings.py --export               # export 50 vectors ke JSON
        """
    )
    parser.add_argument("--vectors", action="store_true", help="Tampilkan sample vectors")
    parser.add_argument("--n", type=int, default=5, help="Jumlah sample vectors (default: 5)")
    parser.add_argument("--search", type=str, help="Query untuk semantic search")
    parser.add_argument("--export", action="store_true", help="Export vectors ke JSON")
    parser.add_argument("--export-limit", type=int, default=50, help="Jumlah maks export (default: 50)")
    parser.add_argument("--url", type=str, default=QDRANT_URL, help=f"Qdrant URL (default: {QDRANT_URL})")
    args = parser.parse_args()

    client = get_client()

    # Selalu tampilkan info collection
    show_collection_info(client)

    if args.vectors:
        show_sample_vectors(client, n=args.n)

    if args.search:
        search_by_query(client, query=args.search)

    if args.export:
        export_vectors(client, limit=args.export_limit)

    if not any([args.vectors, args.search, args.export]):
        print("\n💡 Tip: Tambahkan flag untuk detail lebih:")
        print("   --vectors         → lihat sample vektor")
        print("   --search 'query'  → semantic search")
        print("   --export          → export ke JSON")


if __name__ == "__main__":
    main()
