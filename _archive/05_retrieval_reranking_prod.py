# ============================================================
# 05_retrieval_reranking_prod.py — Stage E: Two-Stage Retrieval (Production)
# Social Welfare Policy Recommender System (Tim 4)
#
# Pipeline: Semantic Search (Qdrant) → Cross-Encoder Reranking
# Versi ini menggunakan HuggingFaceEmbeddings (SentenceTransformers)
# yang berjalan aslinya di VRAM untuk lingkungan produksi (VM RTX 5090).
# ============================================================

import time
import logging
from dataclasses import dataclass, field

# IMPORT CONFIG FIRST to ensure HF_HOME (D: drive) is set before HuggingFace loads!
from config import (
    QDRANT_URL, QDRANT_COLLECTION,
    EMBED_MODEL_NAME, EMBED_BATCH_SIZE,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
)

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASS — Hasil Retrieval
# ============================================================

@dataclass
class RetrievalResult:
    """Satu chunk hasil retrieval beserta skor dan metadata."""
    text: str
    score: float
    embed_score: float = 0.0   # Skor cosine similarity (Stage 1)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "embed_score": round(self.embed_score, 4),
            "metadata": self.metadata,
        }


# ============================================================
# POLICY RETRIEVER CLASS
# ============================================================

class PolicyRetriever:
    """
    Two-Stage Retrieval Pipeline untuk kebijakan sosial.

    Stage 1: Semantic search via Qdrant (embedding similarity).
    Stage 2: Cross-encoder reranking.
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION,
        embed_model_name: str = EMBED_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
        default_top_k: int = RETRIEVAL_TOP_K,
        default_top_n: int = RERANK_TOP_N,
    ):
        self.collection_name = collection_name
        self.default_top_k = default_top_k
        self.default_top_n = default_top_n

        # ── Deteksi device ────────────────────────────────
        import torch
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self._device == "cpu":
            logger.warning(
                "⚠️ CUDA tidak tersedia — menggunakan CPU. "
                "Retrieval tetap berjalan, namun lebih lambat."
            )
        else:
            logger.info("🖥️ Menggunakan GPU (CUDA).")

        # ── Stage 1: Embedding model (HuggingFace) ───────
        logger.info("📦 Memuat embedding model: %s ...", embed_model_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": self._device},
            encode_kwargs={
                "batch_size": EMBED_BATCH_SIZE,
                "normalize_embeddings": True,
            },
        )
        logger.info("✅ Embedding model siap.")

        # ── Stage 2: Cross-Encoder reranker ──────────────
        logger.info("📦 Memuat reranker model: %s ...", reranker_model_name)
        self.reranker = CrossEncoder(
            reranker_model_name,
            device=self._device,
        )
        logger.info("✅ Reranker model siap.")

        # ── Qdrant client ────────────────────────────────
        logger.info("📦 Menghubungkan ke Qdrant server (%s) ...", qdrant_url)
        self.client = QdrantClient(url=qdrant_url)

        # Verifikasi collection
        try:
            info = self.client.get_collection(self.collection_name)
            logger.info(
                "✅ Collection '%s' terhubung (%d points).",
                self.collection_name, info.points_count,
            )
        except Exception as e:
            logger.error(
                "❌ Collection '%s' tidak ditemukan: %s",
                self.collection_name, e,
            )
            raise

    # ──────────────────────────────────────────────────────
    # STAGE 1: Semantic Search
    # ──────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Embed query → cari top-k vektor terdekat di Qdrant.
        """
        top_k = top_k or self.default_top_k

        # Embed query (HuggingFaceEmbeddings)
        query_vector = self.embeddings.embed_query(query)

        # Search Qdrant
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        if not hits:
            logger.warning("⚠️ Semantic search: tidak ada hasil untuk query.")
            return []

        results = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.pop("text", "")
            results.append(RetrievalResult(
                text=text,
                score=hit.score,
                metadata=payload,
            ))

        logger.info(
            "🔍 Semantic search: %d kandidat ditemukan (top score=%.4f).",
            len(results), results[0].score if results else 0,
        )
        return results

    # ──────────────────────────────────────────────────────
    # STAGE 2: Cross-Encoder Reranking
    # ──────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Re-score kandidat menggunakan cross-encoder.
        """
        top_n = top_n or self.default_top_n

        if not candidates:
            logger.warning("⚠️ Reranking: tidak ada kandidat untuk di-rerank.")
            return []

        # Siapkan pasangan (query, text)
        pairs = [(query, c.text) for c in candidates]

        # Prediksi skor relevansi
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.embed_score = candidate.score
            candidate.score = float(score)

        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        finalists = reranked[:top_n]

        logger.info(
            "🏆 Reranking: %d → %d finalis (top score=%.4f).",
            len(candidates), len(finalists),
            finalists[0].score if finalists else 0,
        )
        return finalists

    # ──────────────────────────────────────────────────────
    # FULL PIPELINE
    # ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Pipeline lengkap: Semantic Search → Reranking → Finalis.
        """
        start = time.time()

        candidates = self.semantic_search(query, top_k=top_k)
        if not candidates:
            return []

        finalists = self.rerank(query, candidates, top_n=top_n)

        elapsed = time.time() - start
        logger.info(
            "⏱️ Retrieval selesai dalam %.2f detik (%d finalis).",
            elapsed, len(finalists),
        )
        return finalists


# ============================================================
# MAIN — CLI Testing
# ============================================================

def main():
    print("=" * 65)
    print("🚀 STAGE E: Two-Stage Retrieval (VERSI PRODUKSI / HUGGINGFACE)")
    print(f"   Collection  : {QDRANT_COLLECTION}")
    print(f"   Embed Model : {EMBED_MODEL_NAME}")
    print(f"   Reranker    : {RERANKER_MODEL_NAME}")
    print(f"   Top-K / Top-N: {RETRIEVAL_TOP_K} / {RERANK_TOP_N}")
    print("=" * 65)

    try:
        retriever = PolicyRetriever()
    except Exception as e:
        print(f"\n❌ Gagal inisialisasi retriever: {e}")
        return

    test_queries = [
        "Kriteria penerima bantuan PKH tahun 2025",
        "Bagaimana mekanisme penyaluran BLT Desa?",
        "Data terpadu kesejahteraan sosial",
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 65}")
        print(f"🔎 Query {q_idx}: \"{query}\"")
        print(f"{'─' * 65}")

        results = retriever.retrieve(query)

        if not results:
            print("   ⚠️ Tidak ada hasil ditemukan.")
            continue

        for i, r in enumerate(results, 1):
            sumber = r.metadata.get("Sumber", "-")
            kategori = r.metadata.get("Kategori", "-")
            konteks = r.metadata.get("Konteks_Lengkap", "-")

            print(f"\n   📄 Hasil #{i} (rerank: {r.score:.4f} | embed: {r.embed_score:.4f})")
            print(f"      Sumber   : {sumber}")
            print(f"      Kategori : {kategori}")
            print(f"      Konteks  : {konteks}")
            print(f"      Teks     : {r.text[:200]}{'...' if len(r.text) > 200 else ''}")

    print(f"\n{'=' * 65}")
    print("✅ Testing retrieval pipeline selesai!")
    print("=" * 65)


if __name__ == "__main__":
    main()
