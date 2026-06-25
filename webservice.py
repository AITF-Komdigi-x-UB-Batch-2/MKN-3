# ============================================================
# webservice.py — FastAPI REST API untuk SIRA RAG System
# Social Welfare Policy Recommender System (Tim 3 Universitas Brawijaya)
#
# Endpoints:
#   POST /recommend             → Profil warga → ranking program bantuan (JSON terstruktur)
#   POST /ask                   → Tanya juknis bebas (JSON terstruktur)
#   POST /retrieve              → Retrieval-only tanpa LLM generation
#   GET  /health                → Status Qdrant + model
#
# Jalankan:
#   uvicorn webservice:app --host 0.0.0.0 --port 8002 --reload
#
# Swagger UI:
#   http://localhost:8002/docs
# ============================================================

import os
import re
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
from config import (
    QDRANT_URL, QDRANT_COLLECTION,
    EMBED_MODEL_NAME,
    SYSTEM_PROMPT,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
    RUNPOD_MODEL_NAME,
    configure_utf8_stdio,
)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
configure_utf8_stdio()

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from retrieval import PolicyRetriever, RetrievalResult
from generation import (
    PROGRAM_LABELS,
    build_context_grouped,
    build_context_flat,
)

# ============================================================
# LOGGING — matikan log berisik dari library
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

for noisy in ["httpx", "httpcore", "sentence_transformers", "transformers"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# System prompt is imported from config.py (RANKING_SYSTEM_PROMPT)

JSON_ASK_SYSTEM_PROMPT = """Anda adalah SIRA (Sistem Rekomender Intervensi & Kebijakan Program Sosial), \
asisten pakar kebijakan sosial milik Tim 3 Universitas Brawijaya.

Jawab pertanyaan berdasarkan KONTEKS DOKUMEN yang disediakan.

Anda WAJIB merespons HANYA dengan JSON valid berikut, tanpa teks apapun di luar JSON:

{
  "jawaban": "Jawaban lengkap dan detail berdasarkan dokumen",
  "sumber_digunakan": ["nama dokumen 1", "nama dokumen 2"],
  "poin_penting": ["poin 1", "poin 2", "poin 3"],
  "catatan": "Informasi tambahan atau keterbatasan jawaban jika ada, atau null"
}
"""


# ============================================================
# RETRIEVE SYSTEM PROMPT
# Prefix konteks yang digabungkan ke query retrieval di endpoint /retrieve.
# Ganti isi prompt ini kapan saja untuk mengubah fokus pencarian dokumen
# tanpa perlu mengubah input API.
# ============================================================

RETRIEVE_SYSTEM_PROMPT = (
    "Temukan syarat kelayakan, kriteria sasaran penerima, besaran nominal bantuan, "
    "dan mekanisme pencairan untuk program PKH Plus (lanjut usia 70 tahun ke atas) "
    "dan ASPD (penyandang disabilitas) berdasarkan petunjuk teknis resmi."
)


# ============================================================
# SINGLETON STATE
# ============================================================

class AppState:
    retriever: Optional[object] = None
    llm: Optional[object] = None
    ready: bool = False
    startup_error: Optional[str] = None
    startup_time: Optional[float] = None

state = AppState()


# ============================================================
# LIFESPAN — load model sekali saat startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 SIRA RAG Service starting up...")
    t0 = time.time()
    try:
        state.retriever = PolicyRetriever()
        state.llm = None
        
        # Register local LLM to llm_client
        import llm_client
        llm_client.local_llm = state.llm
        
        state.ready = True
        state.startup_time = round(time.time() - t0, 2)
        
        # Dapatkan IP publik secara opsional untuk visualisasi akses eksternal di VPS
        ip = "[IP_ADDRESS]"
        port = "8002"
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get("https://api.ipify.org")
                if resp.status_code == 200:
                    ip = resp.text.strip()
        except Exception:
            pass

        if ip != "0.0.0.0":
            logger.info(f"✅ Retrieval siap dalam {state.startup_time:.2f}s. Service dapat diakses secara eksternal di: http://{ip}:{port}")
            logger.info(f"✅ Swagger siap diakses di: http://{ip}:{port}/docs")
        else:
            logger.info(f"✅ Retrieval siap dalam {state.startup_time:.2f}s. Generation memakai API RunPod.")
    except Exception as e:
        state.ready = False
        state.startup_error = str(e)
        logger.error("❌ Gagal inisialisasi: %s", e)

    yield

    logger.info("👋 SIRA RAG Service shutting down.")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SIRA RAG API",
    description=(
        "**SIRA — Sistem Rekomender Intervensi & Kebijakan Program Sosial**\n\n"
        "REST API untuk sistem rekomendasi program bantuan sosial berbasis RAG.\n\n"
        "Pipeline: `Qdrant Semantic Search` → `RunPod/OpenAI-compatible LLM Generation`\n\n"
        "MKN 3 Universitas Brawijaya (AITF 2026)"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# SCHEMAS, HELPERS, LLM, AND GUARDRAILS IMPORTS
# ============================================================

from schemas import (
    RecommendRequest, RetrieveOnlyRequest,
    RekomendasiProgram, ProgramTidakSesuai,
    RetrieveChunkResult, RetrieveOnlyResponse,
    RecommendResponse, HealthResponse, ProgramInfo
)

from helpers import (
    _parse_content_to_retrieval_query,
    normalize_spesifikasi,
    to_source_docs,
    infer_retrieval_sources_from_profile,
    retrieval_prompt_for_sources,
    is_profile_query,
)

from llm_client import (
    call_generation_api_checked,
    raise_if_parse_error,
)

from guardrails import (
    enforce_program_eligibility_rules,
    build_fallback_generation,
)


# ============================================================
# HELPERS (WRAPPER)
# ============================================================

def check_ready(require_llm: bool = False):
    if not state.ready or state.retriever is None or (require_llm and state.llm is None):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service belum siap. Model sedang loading atau gagal inisialisasi.",
                "startup_error": state.startup_error,
            }
        )

def retrieve_semantic_only(
    query: str,
    top_k: int,
    top_n: int,
    allowed_sources: Optional[list[str]] = None,
) -> list[RetrievalResult]:
    """
    Wrapper untuk menjalankan semantic search melalui PolicyRetriever.
    """
    limit = max(top_k, top_n)
    
    # Memanggil retrieve dari PolicyRetriever yang sudah kita perbaiki
    results = state.retriever.retrieve(
        query=query,
        top_k=limit,
        allowed_sources=allowed_sources
    )
    
    # Kembalikan dipotong sesuai top_n
    return results[:top_n]


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"],
         summary="Cek status service dan konfigurasi")
def health():
    """
    - `status: ok` → semua model loaded, siap digunakan
    - `status: error` → model gagal load saat startup, cek `startup_error`
    """
    return HealthResponse(
        status="ok" if state.ready else "error",
        ready=state.ready,
        startup_time_s=state.startup_time,
        startup_error=state.startup_error,
        config={
            "qdrant_url": QDRANT_URL,
            "collection": QDRANT_COLLECTION,
            "embed_model": EMBED_MODEL_NAME,
            "llm_model": RUNPOD_MODEL_NAME,
            "default_top_k": RETRIEVAL_TOP_K,
            "default_top_n": RERANK_TOP_N,
        }
    )


@app.get("/programs", response_model=list[ProgramInfo], tags=["System"],
         summary="Daftar program bantuan sosial dalam knowledge base")
def list_programs():
    """Kembalikan daftar program bantuan sosial yang tersedia."""
    return [
        ProgramInfo(filename=f, nama_program=n)
        for f, n in PROGRAM_LABELS.items()
    ]

@app.post("/recommend", response_model=RecommendResponse, tags=["RAG"],
          summary="Rekomendasi program bantuan berdasarkan profil warga")
def recommend(req: RecommendRequest):
    """
    **Mode 1: Profil Warga → Ranking Program Bantuan (JSON Terstruktur)**

    Input profil warga → retrieve juknis relevan → LLM evaluasi eligibilitas → return JSON ranking.

    **Response mencakup:**
    - `rekomendasi[]` — ranking program ELIGIBLE/MUNGKIN_ELIGIBLE + spesifikasi lengkap
    - `program_tidak_sesuai[]` — program yang tidak cocok + alasan
    - `sources[]` — dokumen yang dipakai sebagai konteks (untuk audit/transparansi)
    """
    check_ready()
    t0 = time.time()

    try:
        top_k = req.top_k or RETRIEVAL_TOP_K
        top_n = req.top_n or top_k

        # Deteksi jika profil_warga sudah memuat struktur template prompt lengkap
        core_profile = req.profil_warga
        instructions = None
        has_template = "=== PROFIL WARGA ===" in req.profil_warga
        
        if has_template:
            profile_match = re.search(r"=== PROFIL WARGA ===\s*(.*?)\s*=== AKHIR PROFIL WARGA ===", req.profil_warga, re.DOTALL)
            if profile_match:
                core_profile = profile_match.group(1).strip()
            
            instruction_match = re.search(r"INSTRUKSI EKSEKUSI:\s*(.*)", req.profil_warga, re.DOTALL)
            if instruction_match:
                instructions = instruction_match.group(1).strip()

        query_for_retrieval = _parse_content_to_retrieval_query(core_profile)
        inferred_sources = infer_retrieval_sources_from_profile(core_profile)

        # ── Prepend prompt retrieval sesuai program target untuk meningkatkan kemiripan nominal ──
        retrieval_prompt = retrieval_prompt_for_sources(inferred_sources)
        if retrieval_prompt and retrieval_prompt.strip():
            query_for_retrieval = f"{retrieval_prompt.strip()}\n\n{query_for_retrieval}"

        logger.info("🔎 /recommend effective_query: %s", query_for_retrieval[:200])
        if inferred_sources:
            logger.info("🎯 /recommend target program sources: %s", inferred_sources)

        results = retrieve_semantic_only(
            query_for_retrieval,
            top_k=top_k,
            top_n=top_n,
            allowed_sources=inferred_sources or list(PROGRAM_LABELS.keys()),   # [Bug 3 Fix]
        )

        if not results:
            raise HTTPException(status_code=404,
                detail="Tidak ada dokumen 6 program utama yang relevan ditemukan untuk profil yang diberikan.")

        # ── Jalankan Metode 1: Sisipkan chunk nominal untuk menjamin ketersediaan data nominal ──
        target_sources_for_nominal = inferred_sources or list(set(
            r.metadata.get("sumber") for r in results 
            if r.metadata.get("sumber") in PROGRAM_LABELS
        ))
        
        nominal_chunks = []
        if target_sources_for_nominal:
            for src_file in target_sources_for_nominal:
                try:
                    nom_hits = state.retriever.retrieve_nominal_chunk(src_file)
                    nominal_chunks.extend(nom_hits)
                except Exception as ex:
                    logger.warning("⚠️ Gagal mengambil nominal chunk untuk %s: %s", src_file, ex)

        if nominal_chunks:
            seen_texts = {r.text for r in results}
            added_count = 0
            for nom_chunk in nominal_chunks:
                if nom_chunk.text not in seen_texts:
                    results.append(nom_chunk)
                    seen_texts.add(nom_chunk.text)
                    added_count += 1
            if added_count > 0:
                logger.info("💉 Berhasil menyisipkan %d chunk nominal tambahan ke hasil retrieval.", added_count)

        context = build_context_grouped(results)

        if has_template:
            user_prompt = f"=== PROFIL WARGA ===\n{core_profile}\n=== AKHIR PROFIL WARGA ===\n\n"
            user_prompt += f"=== KONTEKS DOKUMEN KEBIJAKAN DARI RETRIEVAL ===\n{context}\n=== AKHIR KONTEKS DOKUMEN ===\n\n"
            if instructions:
                user_prompt += f"INSTRUKSI EKSEKUSI:\n{instructions}"
            else:
                user_prompt += (
                    "INSTRUKSI EKSEKUSI:\n"
                    "1. Lakukan audit kelayakan secara objektif dengan mencocokkan kriteria pada Profil Warga terhadap aturan di Konteks Dokumen.\n"
                    "2. Hasilkan output TEPAT dalam format Toon dengan empat kategori wajib: 'ringkasan_profil', 'rekomendasi', 'rekomendasi_teknis_bansos', dan 'program_tidak_sesuai'.\n"
                    "3. Pada baris 'rekomendasi', Anda wajib memuat informasi: rank, dasar hukum, dan alasan kelayakan di dalam kolom Detail/Alasan.\n"
                    "4. Pada baris 'rekomendasi_teknis_bansos', jabarkan rencana aksi operasional, prioritas pemanfaatan dana, mekanisme pendampingan, pengelola bantuan, serta monitoring evaluasi warga di lapangan.\n"
                    "5. Program yang tidak cocok harus masuk program_tidak_sesuai dengan alasan spesifik.\n"
                    "6. Respons hanya berupa teks format Toon valid. Jangan memakai markdown, heading, atau placeholder."
                )
        else:
            user_prompt = (
                "=== PROFIL WARGA ===\n"
                f"{req.profil_warga}\n"
                "=== AKHIR PROFIL WARGA ===\n\n"
                "=== KONTEKS DOKUMEN KEBIJAKAN DARI RETRIEVAL ===\n"
                f"{context}\n"
                "=== AKHIR KONTEKS DOKUMEN ===\n\n"
                "INSTRUKSI EKSEKUSI:\n"
                "1. Lakukan audit kelayakan secara objektif dengan mencocokkan kriteria pada Profil Warga terhadap aturan di Konteks Dokumen.\n"
                "2. Hasilkan output TEPAT dalam format Toon dengan empat kategori wajib: 'ringkasan_profil', 'rekomendasi', 'rekomendasi_teknis_bansos', dan 'program_tidak_sesuai'.\n"
                "3. Pada baris 'rekomendasi', Anda wajib memuat informasi: rank, dasar hukum, dan alasan kelayakan di dalam kolom Detail/Alasan.\n"
                "4. Pada baris 'rekomendasi_teknis_bansos', jabarkan rencana aksi operasional, prioritas pemanfaatan dana, mekanisme pendampingan, pengelola bantuan, serta monitoring evaluasi warga di lapangan.\n"
                "5. Program yang tidak cocok harus masuk program_tidak_sesuai dengan alasan spesifik.\n"
                "6. Respons hanya berupa teks format Toon valid. Jangan memakai markdown, heading, atau placeholder.\n"
            )

        generation_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            parsed = call_generation_api_checked(generation_messages)
        except HTTPException as e:
            logger.warning(
                "⚠️ Generation API gagal menghasilkan JSON valid, fallback deterministic dipakai: %s",
                e.detail,
            )
            parsed = build_fallback_generation(core_profile, results)
        except Exception as e:
            logger.warning(
                "⚠️ Generation API Tim 1/RunPod gagal, fallback deterministic dipakai: %s",
                e,
            )
            parsed = build_fallback_generation(core_profile, results)
        raise_if_parse_error(parsed)
        logger.info("DEBUG PARSED: %s", json.dumps(parsed, indent=2, ensure_ascii=False))
        parsed = enforce_program_eligibility_rules(parsed, core_profile)

        elapsed_ms = int((time.time() - t0) * 1000)
        program_count = len(set(r.metadata.get("sumber", "") for r in results))

        rekomendasi = [
            RekomendasiProgram(
                rank=item.get("rank", i + 1),
                nama_program=item.get("nama_program", ""),
                status=item.get("status", ""),
                sumber=item.get("sumber") or item.get("dasar_hukum"),
                alasan_kelayakan=item.get("alasan_kelayakan"),
            )
            for i, item in enumerate(parsed.get("rekomendasi", []))
        ]

        tidak_sesuai = [
            ProgramTidakSesuai(
                nama_program=item.get("nama_program", ""),
                status=item.get("status", "TIDAK_ELIGIBLE"),
                alasan=item.get("alasan"),
            )
            for item in parsed.get("program_tidak_sesuai", [])
        ]

        rekomendasi_teknis = parsed.get("rekomendasi_teknis_bansos")
        if isinstance(rekomendasi_teknis, dict):
            parts = []
            for k, v in rekomendasi_teknis.items():
                k_clean = k.replace("_", " ").title()
                if isinstance(v, list):
                    v_str = ", ".join(v)
                else:
                    v_str = str(v)
                parts.append(f"{k_clean}: {v_str}")
            rekomendasi_teknis = "; ".join(parts)
        elif rekomendasi_teknis is not None:
            rekomendasi_teknis = str(rekomendasi_teknis).strip()

        return RecommendResponse(
            ringkasan_profil=parsed.get("ringkasan_profil", ""),
            rekomendasi=rekomendasi,
            rekomendasi_teknis_bansos=rekomendasi_teknis,
            program_tidak_sesuai=tidak_sesuai,
            sources=to_source_docs(results),
            retrieval_count=len(results),
            program_count=program_count,
            elapsed_ms=elapsed_ms,
            model_used=RUNPOD_MODEL_NAME,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ /recommend error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/retrieve", response_model=RetrieveOnlyResponse, tags=["Retrieval"],
          summary="Ambil hasil retrieval semantic search tanpa LLM generation")
def retrieve_only(req: RetrieveOnlyRequest):
    """
    **Retrieval-Only: Semantic Search (tanpa LLM)**

    Endpoint ini menjalankan semantic search:
    1. **Semantic Search** — embed query → cari top-K vektor terdekat di Qdrant
    2. **Finalisasi** — return top-N terbaik berdasarkan skor embedding

    **Tidak ada** LLM yang dipanggil. Cocok untuk:
    - Inspeksi konteks sebelum dikirim ke LLM
    - Debugging pipeline retrieval
    - Mengambil konteks chunk untuk diproses sendiri

    **Field `filter_programs_only`** — jika true, filter dilakukan secara
    native di Qdrant (bukan post-filter Python) via `allowed_sources`.
    """
    check_ready()
    t0 = time.time()

    try:
        top_k = req.top_k or RETRIEVAL_TOP_K
        top_n = req.top_n or top_k

        # ── Step 1: Detect if content is a profile query ────────────────────
        is_profile = is_profile_query(req.content)

        if is_profile:
            # ── Step 1a: Parse content → keyword query padat ─────────────────────
            effective_query = _parse_content_to_retrieval_query(req.content)
            inferred_sources = infer_retrieval_sources_from_profile(req.content)

            # ── Step 2: Gabungkan prompt retrieval sesuai program target ────────
            retrieval_prompt = retrieval_prompt_for_sources(inferred_sources)
            if retrieval_prompt and retrieval_prompt.strip():
                effective_query = f"{retrieval_prompt.strip()}\n\n{effective_query}"
        else:
            # Jika merupakan kueri pencarian bebas (ad-hoc), gunakan kueri asli apa adanya
            effective_query = req.content.strip()
            inferred_sources = None

        logger.info("🔎 /retrieve effective_query (200 chars): %s", effective_query[:200])

        # ── Step 3: Jalankan retrieval ──────────────────────────────────────
        allowed = (
            inferred_sources
            if inferred_sources
            else list(PROGRAM_LABELS.keys()) if req.filter_programs_only else None
        )
        if inferred_sources:
            logger.info("🎯 /retrieve target program sources: %s", inferred_sources)

        results = retrieve_semantic_only(
            effective_query,
            top_k=top_k,
            top_n=top_n,
            allowed_sources=allowed, 
        )

        if not results:
            detail = (
                "Tidak ada chunk dari program yang relevan untuk query tersebut."
                if req.filter_programs_only
                else "Tidak ada dokumen relevan ditemukan untuk query tersebut."
            )
            raise HTTPException(status_code=404, detail=detail)

        programs_covered = len(set(
            r.metadata.get("sumber", "") for r in results
            if r.metadata.get("sumber", "") in PROGRAM_LABELS
        ))

        elapsed_ms = int((time.time() - t0) * 1000)

        chunk_results = [
            RetrieveChunkResult(
                rank=idx,
                program=PROGRAM_LABELS.get(
                    r.metadata.get("sumber", ""),
                    r.metadata.get("sumber", "").replace(".pdf", "")
                ),
                sumber=r.metadata.get("sumber", "unknown"),
                judul_halaman=r.metadata.get("judul_halaman"),
                page_number=str(r.metadata.get("page_number", "")),
                rerank_score=round(r.score, 4),
                embed_score=round(r.embed_score, 4),
                text=r.text,
                metadata=r.metadata,
            )
            for idx, r in enumerate(results, 1)
        ]

        return RetrieveOnlyResponse(
            retrieval_count=len(chunk_results),
            programs_covered=programs_covered,
            elapsed_ms=elapsed_ms,
            results=chunk_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ /retrieve error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    configure_utf8_stdio()
    import uvicorn
    uvicorn.run(
        "webservice:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )