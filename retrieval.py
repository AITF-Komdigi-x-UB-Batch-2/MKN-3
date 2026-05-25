# ============================================================
# retrieval.py — Stage E: Two-Stage Retrieval Pipeline
# Social Welfare Policy Recommender System (Tim 4)
#
# Pipeline: Semantic Search (Qdrant) → Cross-Encoder Reranking
# Dirancang agar dapat di-import ke FastAPI / webservice.
#
# Fix v2:
#   - Device detection fix: platform check sebelum MPS agar
#     Windows + CUDA tidak salah masuk ke MPS (CPU fallback)
#   - import re sudah ada untuk strip prefix
# ============================================================

import time
import sys
import logging
import re
import platform
from typing import Any
from dataclasses import dataclass, field

from config import (
    QDRANT_URL, QDRANT_COLLECTION,
    EMBED_MODEL_NAME, EMBED_DIMENSIONS,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
)

from qdrant_client import QdrantClient
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
# DEVICE DETECTION (Fix: platform check untuk Windows vs Mac)
# ============================================================

def detect_device() -> str:
    """
    Detect best available device dengan benar untuk semua platform.
    
    Masalah sebelumnya:
      torch.backends.mps.is_available() bisa return True di Mac tapi
      CrossEncoder tidak support MPS dengan baik → fallback ke CPU yang lambat.
      Di Windows, MPS tidak ada tapi urutan check yang salah bisa menyebabkan
      device salah di-set.
    
    Fix:
      - MPS hanya dipakai di macOS (Darwin)
      - Windows/Linux langsung ke CUDA check
    """
    import torch
    system = platform.system()
    
    if system == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class RetrievalResult:
    """Satu chunk hasil retrieval beserta skor dan metadata."""
    text: str
    score: float
    embed_score: float = 0.0
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

    Stage 1: Semantic search via Qdrant.
    Stage 2: Cross-encoder reranking (BAAI/bge-reranker-v2-m3).

    Usage:
        retriever = PolicyRetriever()
        results   = retriever.retrieve("Kriteria penerima PKH")
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
        self.embed_model_name = embed_model_name

        # ── Device detection (fixed) ──────────────────────
        self._device = detect_device()

        if self._device == "cpu":
            logger.warning(
                "⚠️ GPU tidak tersedia — menggunakan CPU. "
                "Retrieval dan reranking akan lebih lambat."
            )
        else:
            logger.info("🖥️ Menggunakan device: %s", self._device.upper())

        # ── Embedding model ───────────────────────────────
        logger.info("📦 Inisialisasi Embedding: %s (%dd)", embed_model_name, EMBED_DIMENSIONS)
        self.embeddings = self._init_embedding_model(embed_model_name)

        # ── Reranker ──────────────────────────────────────
        logger.info("📦 Memuat reranker: %s (device=%s)...", reranker_model_name, self._device)
        self.reranker = CrossEncoder(
            reranker_model_name,
            device=self._device,
        )
        logger.info("✅ Reranker siap.")

        # ── Qdrant client ────────────────────────────────
        logger.info("📦 Menghubungkan ke Qdrant (%s)...", qdrant_url)
        self.client = QdrantClient(url=qdrant_url)

        try:
            info = self.client.get_collection(self.collection_name)
            logger.info(
                "✅ Collection '%s' terhubung (%d points).",
                self.collection_name, info.points_count,
            )
        except Exception as e:
            logger.error("❌ Collection '%s' tidak ditemukan: %s", self.collection_name, e)
            raise

    def _init_embedding_model(self, embed_model_name: str):
        """Inisialisasi embedding: Ollama atau HuggingFace."""
        is_ollama = (
            ":" in embed_model_name
            or embed_model_name == "embeddinggemma"
            or "ollama" in embed_model_name.lower()
        )

        if is_ollama:
            try:
                from langchain_ollama import OllamaEmbeddings
                from config import OLLAMA_BASE_URL
                embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=embed_model_name)
                embeddings.embed_query("test")
                logger.info("✅ Ollama embedding siap.")
                return embeddings
            except Exception as e:
                logger.warning("⚠️ Ollama embedding gagal (%s). Fallback ke HuggingFace...", e)

        import torch
        device = self._device
        logger.info("🖥️ HuggingFace embedding device: %s", device)

        model_kwargs: dict = {"device": device}

        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✅ HuggingFace embedding siap (device=%s).", device)
        return embeddings

    # ──────────────────────────────────────────────────────
    # STAGE 1: Semantic Search
    # ──────────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int | None = None,
        query_filter: Any | None = None,
    ) -> list[RetrievalResult]:
        """
        Embed query → cari top-k vektor terdekat di Qdrant.
        Strip prefix [Sumber:...] dari teks sebelum return agar
        teks yang disimpan di RetrievalResult bersih untuk display & LLM.
        Prefix asli disimpan di metadata['context_prefix'].
        """
        top_k = top_k or self.default_top_k
        query_vector = self.embeddings.embed_query(query)

        if len(query_vector) != EMBED_DIMENSIONS:
            logger.warning(
                "⚠️ Dimensi embedding (%d) ≠ config (%d).",
                len(query_vector), EMBED_DIMENSIONS,
            )

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        if not hits:
            logger.warning("⚠️ Semantic search: tidak ada hasil.")
            return []

        results = []
        for hit in hits:
            payload = hit.payload or {}
            raw_text = payload.pop("text", "")

            # Strip prefix [Sumber: ... | Hal. N | Judul]
            prefix_match = re.match(r'^(\[Sumber:[^\]]+\])\n?', raw_text)
            context_prefix = prefix_match.group(1) if prefix_match else ""
            text_clean = raw_text[len(context_prefix):].lstrip("\n").strip()

            results.append(RetrievalResult(
                text=text_clean,
                score=hit.score,
                metadata={**payload, "context_prefix": context_prefix},
            ))

        logger.info(
            "🔍 Semantic search: %d kandidat (top score=%.4f).",
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
        """Re-score kandidat dengan cross-encoder, return top_n terbaik."""
        top_n = top_n or self.default_top_n

        if not candidates:
            logger.warning("⚠️ Reranking: tidak ada kandidat.")
            return []

        pairs = [(query, c.text) for c in candidates]
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
        query_filter: Any | None = None,
    ) -> list[RetrievalResult]:
        """Pipeline lengkap: Semantic Search → Reranking → Finalis."""
        start = time.time()

        candidates = self.semantic_search(query, top_k=top_k, query_filter=query_filter)
        if not candidates:
            return []

        finalists = self.rerank(query, candidates, top_n=top_n)

        elapsed = time.time() - start
        logger.info("⏱️ Retrieval selesai: %.2f detik (%d finalis).", elapsed, len(finalists))
        return finalists


# ============================================================
# MAIN — CLI Testing
# ============================================================

def main():
    print("=" * 65)
    print("🚀 RETRIEVAL — Two-Stage (BGE-M3 + BGE-Reranker-v2-M3)")
    print(f"   Collection  : {QDRANT_COLLECTION}")
    print(f"   Embed Model : {EMBED_MODEL_NAME} ({EMBED_DIMENSIONS}d)")
    print(f"   Reranker    : {RERANKER_MODEL_NAME}")
    print(f"   Top-K/Top-N : {RETRIEVAL_TOP_K}/{RERANK_TOP_N}")
    print(f"   Platform    : {platform.system()} | Device: {detect_device().upper()}")
    print("=" * 65)

    try:
        retriever = PolicyRetriever()
    except Exception as e:
        print(f"\n❌ Gagal inisialisasi retriever: {e}")
        sys.exit(1)

    test_queries = [
        "Siapa saja sasaran penerima bantuan sosial PKH Plus Jawa Timur 2026?",
        "Apa syarat penyandang disabilitas untuk mendapatkan bantuan ASPD?",
        "Bagaimana mekanisme penyaluran bantuan penanganan kemiskinan ekstrem?",
        "Apa kriteria KPM yang bisa mengajukan KIP KPM JAWARA?",
        "Bagaimana prosedur pengajuan bantuan KIP Putri JAWARA untuk perempuan?",
        "Apa peran pendamping dalam verifikasi dan validasi data penerima manfaat?",
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 65}")
        print(f"🔎 Query {q_idx}: \"{query}\"")
        print(f"{'─' * 65}")

        results = retriever.retrieve(query)
        if not results:
            print("   ⚠️ Tidak ada hasil.")
            continue

        for i, r in enumerate(results, 1):
            sumber  = r.metadata.get("sumber", "-")
            judul   = r.metadata.get("judul_halaman", "")
            halaman = r.metadata.get("page_number", "-")
            prefix  = r.metadata.get("context_prefix", "")

            print(f"\n   📄 #{i} (rerank: {r.score:.4f} | embed: {r.embed_score:.4f})")
            print(f"      Sumber  : {sumber} (hal. {halaman})")
            if judul:
                print(f"      Judul   : {judul}")
            if prefix:
                print(f"      Prefix  : {prefix}")
            print(f"      Teks    : {r.text[:200]}{'...' if len(r.text) > 200 else ''}")

    print(f"\n{'=' * 65}")
    print("✅ Testing selesai!")
    print("=" * 65)


if __name__ == "__main__":
    main()