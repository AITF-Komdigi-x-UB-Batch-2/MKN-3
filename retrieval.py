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
import re
import warnings
import logging
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

from qdrant_client import QdrantClient, models

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

    def filter_program(self, query: str) -> str | None:
        query_lower = query.lower()
        
        # Cari skor PKH Plus / PKH+
        pkh_score = 0.0
        pkh_match = re.search(r'(?:skor\s+)?pkh\s*\+?\s*(?:plus)?\s*:\s*([0-9.]+)', query_lower)
        if pkh_match:
            pkh_score = float(pkh_match.group(1))
            
        # Cari skor ASPD
        aspd_score = 0.0
        aspd_match = re.search(r'(?:skor\s+)?aspd\s*:\s*([0-9.]+)', query_lower)
        if aspd_match:
            aspd_score = float(aspd_match.group(1))
            
        if pkh_score == 0.0 and aspd_score == 0.0:
            return None
            
        if pkh_score > aspd_score:
            return "PKH Plus"
        else:
            return "ASPD"

    def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        allowed_sources: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """
        Jalankan retrieval pipeline:
          1. Semantic search ke Qdrant (top_k kandidat)

        Args:
            query: teks query (profil warga / pertanyaan)
            top_k: jumlah kandidat dari embedding search
            allowed_sources: list nilai field 'sumber' yang diizinkan
                             (Qdrant native filter, MatchAny)
        Returns:
            list[RetrievalResult] terurut descending by rerank score
        """
        if not query or not query.strip():
            logger.warning("⚠️ Query kosong, skip retrieval.")
            return []

        # ── Build Qdrant filter ──────────────────────────────
        conditions = []
        if allowed_sources:
            conditions.append(
                models.FieldCondition(
                    key="sumber",
                    match=models.MatchAny(any=allowed_sources),
                )
            )
        else:
            detected_program = self.filter_program(query)
            if detected_program:
                conditions.append(
                    models.FieldCondition(
                        key="nama_bansos",
                        match=models.MatchValue(value=detected_program),
                    )
                )
        qdrant_filter = models.Filter(must=conditions) if conditions else None


        # ── Semantic search ──────────────────────────────────
        logger.debug("🔎 Semantic-only search limit=%d ...", top_k)
        hits = self.client.query_points(
            collection_name=self.collection,
            query=models.Document(text=query, model=config.EMBED_MODEL_NAME),
            using=self.vector_name,
            limit=top_k,
            query_filter=qdrant_filter,
        ).points
        
        if not hits:
            logger.warning("⚠️ Tidak ada hasil dari Qdrant.")
            return []
        
        results = []
        for h in hits:
            payload = h.payload or {}
            text = payload.get("text", "")
            # skema metadata selain text (content)
            metadata = {k: v for k, v in payload.items() if k != "text"}
            results.append(RetrievalResult(
                text=text,
                metadata=metadata,
                embed_score=float(h.score),
            ))
        logger.info(
            "🔎 Semantic search selesai: %d hasil dikembalikan (top score=%.4f).",
            len(results), results[0].score if results else 0.0
        )
        return results