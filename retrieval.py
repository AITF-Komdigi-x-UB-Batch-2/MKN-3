# ============================================================
# retrieval.py — PolicyRetriever: Semantic Search + Reranking
# Social Welfare Policy Recommender System (Tim 3)
#
# Dipakai oleh webservice.py sebagai module:
#   from retrieval import PolicyRetriever, RetrievalResult
#
# Pipeline:
#   1. Embed query via fastembed (QdrantClient.set_model)
#   2. Semantic search ke Qdrant (named vector, filter by sumber)
#   3. Cross-encoder reranking (sentence-transformers)
# ============================================================

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


# ============================================================
# DATACLASS RESULT
# ============================================================

@dataclass
class RetrievalResult:
    """Satu hasil retrieval: teks chunk + metadata + skor."""
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0        # skor reranker (cross-encoder)
    embed_score: float = 0.0  # skor embedding (cosine similarity)


# ============================================================
# POLICY RETRIEVER
# ============================================================

class PolicyRetriever:
    """
    Semantic search + reranking untuk dokumen kebijakan sosial.

    Langkah:
    1. QdrantClient.query_points (fastembed named vector)
    2. CrossEncoder reranking (ms-marco atau bge-reranker)
    """

    def __init__(self):
        logger.info("📦 Inisialisasi PolicyRetriever...")

        # Qdrant client dengan fastembed
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.client.set_model(config.EMBED_MODEL_NAME)
        self.vector_name = list(self.client.get_fastembed_vector_params().keys())[0]
        self.collection  = config.QDRANT_COLLECTION
        logger.info("✅ Qdrant terhubung — collection: %s, vector: %s",
                    self.collection, self.vector_name)

        # Cross-encoder reranker
        logger.info("📦 Memuat reranker: %s ...", config.RERANKER_MODEL_NAME)
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL_NAME,
            max_length=512,
        )
        logger.info("✅ Reranker siap.")

    def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        top_n: int = config.RERANK_TOP_N,
        allowed_sources: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Jalankan retrieval pipeline:
          1. Semantic search ke Qdrant (top_k kandidat)
          2. Cross-encoder reranking → ambil top_n terbaik

        Args:
            query: teks query (profil warga / pertanyaan)
            top_k: jumlah kandidat dari embedding search
            top_n: jumlah hasil final setelah reranking
            allowed_sources: list nilai field 'sumber' yang diizinkan
                             (Qdrant native filter, MatchAny)
        Returns:
            list[RetrievalResult] terurut descending by rerank score
        """
        if not query or not query.strip():
            logger.warning("⚠️ Query kosong, skip retrieval.")
            return []

        # ── Build Qdrant filter ──────────────────────────────
        qdrant_filter = None
        if allowed_sources:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="sumber",
                        match=MatchAny(any=allowed_sources),
                    )
                ]
            )

        # ── Semantic search ──────────────────────────────────
        from qdrant_client import models as qmodels
        logger.debug("🔎 Semantic search top_k=%d ...", top_k)
        hits = self.client.query_points(
            collection_name=self.collection,
            query=qmodels.Document(text=query, model=config.EMBED_MODEL_NAME),
            using=self.vector_name,
            limit=top_k,
            query_filter=qdrant_filter,
        ).points

        if not hits:
            logger.warning("⚠️ Tidak ada hasil dari Qdrant.")
            return []

        # ── Cross-encoder reranking ──────────────────────────
        texts = [h.payload.get("text", "") for h in hits]
        pairs = [(query, t) for t in texts]

        scores = self.reranker.predict(pairs)

        # Gabungkan skor embedding + reranker
        results = [
            RetrievalResult(
                text=texts[i],
                metadata={k: v for k, v in hits[i].payload.items() if k != "text"},
                score=float(scores[i]),
                embed_score=float(hits[i].score),
            )
            for i in range(len(hits))
        ]

        # Urutkan descending by rerank score, ambil top_n
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]