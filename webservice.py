# ============================================================
# webservice.py — FastAPI REST API untuk SIRA RAG System
# Social Welfare Policy Recommender System (Tim 3 Universitas Brawijaya)
#
# Endpoints:
#   POST /recommend              → Profil warga → ranking program bantuan (JSON terstruktur)
#   POST /ask                   → Tanya juknis bebas (JSON terstruktur)
#   POST /retrieve              → Retrieval-only tanpa LLM generation
#   GET  /health                → Status Qdrant + model
#   GET  /programs              → Daftar 6 program bantuan
#
# Jalankan:
#   uvicorn webservice:app --host 0.0.0.0 --port 8000 --reload
#
# Swagger UI:
#   http://localhost:8000/docs
#
# Changelog v3:
#   [Bug 1] Hapus pemotongan [:300] di query_for_retrieval pada endpoint
#           /recommend; gunakan _parse_content_to_retrieval_query() untuk
#           seluruh string profil.
#   [Bug 2] _parse_content_to_retrieval_query: tambahkan safe JSON parsing
#           di awal fungsi sebelum fallback ke Regex.
#   [Bug 3] Pemfilteran PROGRAM_LABELS dipindahkan ke native Qdrant filter
#           (FieldCondition/MatchAny) via parameter allowed_sources di
#           retriever.retrieve(). top_k default dinaikkan ke min 40 agar
#           semua 6 program terwakili di pool retrieval.
#   [Bug 4] Seragamkan nama tim → "Tim 3 Universitas Brawijaya".
# ============================================================

import os
import re
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# PINDAHKAN KE ATAS: Agar HF_HOME di config aktif sebelum library AI lain di-import
from config import (
    QDRANT_URL, QDRANT_COLLECTION,
    EMBED_MODEL_NAME,
    PROMPT_TEMPLATE, POLICY_PROMPT_TEMPLATE,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
    TIM1_CLASSIFICATION_API_URL, TIM1_GENERATION_API_URL, TIM1_API_TIMEOUT_S,
    RUNPOD_API_KEY, RUNPOD_MODEL_NAME, RUNPOD_TEMPERATURE, RUNPOD_MAX_TOKENS,
    configure_utf8_stdio,
)
configure_utf8_stdio()

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

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


# ============================================================
# SYSTEM PROMPTS — Paksa LLM output JSON
# ============================================================

JSON_RANKING_SYSTEM_PROMPT = "Anda adalah AI Auditor resmi Dinas Sosial Provinsi Jawa Timur yang bertugas melakukan verifikasi dan validasi kelayakan penerima manfaat dua program bantuan sosial.\n\nTugas Anda: Berdasarkan PROFIL WARGA dan KONTEKS PROGRAM BANTUAN yang disediakan, evaluasi kelayakan warga HANYA untuk 2 program utama berikut:\n1. Asistensi Sosial Penyandang Disabilitas (ASPD)\n2. PKH Plus (Lanjut Usia 70+)\n\n=== INSTRUKSI PENTING ===\n1. Evaluasi hanya 2 program utama di atas secara individual.\n2. Tentukan status: \"ELIGIBLE\" atau \"TIDAK_ELIGIBLE\".\n3. Ranking dari yang paling cocok ke yang paling tidak cocok.\n4. Berikan reasoning yang jelas dan WAJIB mengutip sumber dokumen resmi juknis.\n5. JANGAN merekomendasikan program bantuan di luar 2 program utama tersebut.\n6. DILARANG KERAS menyebut Program Sembako, PKH reguler, BPNT, PBI Jaminan Kesehatan, Rutilahu, PIP, Jamkesda, atau bantuan tambahan lainnya.\n\n=== FORMAT OUTPUT ===\nAnda WAJIB merespons HANYA dengan JSON valid tanpa markdown dan tanpa teks pembuka/penutup.\nGunakan key berikut dengan urutan persis:\n- ringkasan_profil: string konkret berisi umur, desil, status DTSEN, disabilitas/usia lansia, dan kondisi kunci warga.\n- rekomendasi: array program yang ELIGIBLE atau MUNGKIN_ELIGIBLE. Setiap item di dalamnya wajib berisi key: rank, nama_program, status, dasar_hukum, dan alasan_kelayakan.\n- rekomendasi_teknis_bansos: string narasi tunggal (paragraf utuh tanpa objek/poin berlapis) yang menjabarkan rencana aksi operasional, prioritas pemanfaatan dana, mekanisme pendampingan, pengelola bantuan, serta monitoring evaluasi warga di lapangan. Jika warga tidak berhak menerima program bantuan apa pun (array rekomendasi kosong), maka nilai key ini WAJIB disetel null secara kaku.\n- program_tidak_sesuai: array program yang TIDAK_ELIGIBLE. Setiap item di dalamnya wajib berisi key: nama_program, status, dan alasan.\n\nLarangan keras:\n- Jangan menyalin placeholder seperti \"Nama Program\", \"Rp X.XXX.XXX\", \"dst\", \"rangkuman singkat\", atau \"Penjelasan mengapa\".\n- Jangan mengosongkan alasan. Semua alasan harus merujuk kondisi riil warga dan kriteria dokumen.\n- nama_program harus ditulis persis salah satu dari 2 program utama yang disebut di atas."

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
# Anda bisa mengganti isi RETRIEVE_SYSTEM_PROMPT di atas sesuai kebutuhan.
# Contoh alternatif:
#   RETRIEVE_SYSTEM_PROMPT = "Cari aturan desil kemiskinan dan syarat DTKS untuk bantuan sosial Jawa Timur"
#   RETRIEVE_SYSTEM_PROMPT = "Temukan prosedur pencairan dan dokumen yang diperlukan untuk KIP JAWARA"


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
        vps_ip = "0.0.0.0"
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get("https://api.ipify.org")
                if resp.status_code == 200:
                    vps_ip = resp.text.strip()
        except Exception:
            pass

        if vps_ip != "0.0.0.0":
            logger.info("✅ Retrieval siap dalam %.2fs. Service dapat diakses secara eksternal di: http://%s:8000", state.startup_time, vps_ip)
        else:
            logger.info("✅ Retrieval siap dalam %.2fs. Generation memakai API RunPod.", state.startup_time)
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
        "Tim 3 Universitas Brawijaya × DISKOMINFO Jawa Timur (AITF 2026)"
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
    RecommendRequest, AskRequest, RetrieveOnlyRequest,
    SpesifikasiProgram, RekomendasiProgram, ProgramTidakSesuai,
    SourceDocument, RetrieveChunkResult, RetrieveOnlyResponse,
    RecommendResponse, AskResponse, HealthResponse, ProgramInfo
)

from helpers import (
    _parse_content_to_retrieval_query,
    normalize_spesifikasi,
    to_source_docs,
    infer_retrieval_sources_from_profile,
    retrieval_prompt_for_sources,
)

from llm_client import (
    invoke_llm,
    call_classification_api,
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
            # "reranker": RERANKER_MODEL_NAME,  # Dinonaktifkan untuk uji coba semantic-only.
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
        top_n = req.top_n or RERANK_TOP_N

        # [Bug 1 Fix] Gunakan _parse_content_to_retrieval_query() atas
        # SELURUH string profil_warga tanpa [:300] slicing.
        # BGE-M3 mendukung context window besar (8192 token), sehingga
        # informasi disabilitas, usaha, dll. di bagian bawah profil tidak hilang.
        query_for_retrieval = _parse_content_to_retrieval_query(req.profil_warga)
        inferred_sources = infer_retrieval_sources_from_profile(req.profil_warga)

        logger.info("🔎 /recommend effective_query: %s", query_for_retrieval[:200])
        if inferred_sources:
            logger.info("🎯 /recommend target program sources: %s", inferred_sources)

        # [Bug 3 Fix] Filter 6 program utama dilakukan secara native di Qdrant
        # via allowed_sources — bukan post-filter list comprehension.
        # Dengan ini kuota top_k tidak terbuang oleh dokumen luar program.
        # RERANKER OFF: jalur lama memakai Cross-Encoder reranking.
        # results = state.retriever.retrieve(
        #     query_for_retrieval,
        #     top_k=top_k,
        #     top_n=top_n,
        #     allowed_sources=list(PROGRAM_LABELS.keys()),   # [Bug 3 Fix]
        # )
        results = retrieve_semantic_only(
            query_for_retrieval,
            top_k=top_k,
            top_n=top_n,
            allowed_sources=inferred_sources or list(PROGRAM_LABELS.keys()),   # [Bug 3 Fix]
        )

        if not results:
            raise HTTPException(status_code=404,
                detail="Tidak ada dokumen 6 program utama yang relevan ditemukan untuk profil yang diberikan.")

        context = build_context_grouped(results)

        profil_section = f"=== PROFIL WARGA ===\n{req.profil_warga}"

        user_prompt = (
            "=== PROFIL WARGA DARI TIM 4 (ACUAN UTAMA) ===\n"
            f"{req.profil_warga}\n"
            "=== AKHIR PROFIL WARGA ===\n\n"
            "=== KONTEKS DOKUMEN KEBIJAKAN DARI RETRIEVAL ===\n"
            f"{context}\n"
            "=== AKHIR KONTEKS DOKUMEN ===\n\n"
            "INSTRUKSI EKSEKUSI:\n"
            "1. Isi JSON dengan data konkret dari profil warga dan konteks dokumen.\n"
            "2. Jika warga lansia 70+ dan desil/DTSEN memenuhi, prioritaskan evaluasi PKH Plus.\n"
            "3. Jika umur warga kurang dari 70 tahun, PKH Plus wajib TIDAK_ELIGIBLE.\n"
            "4. Jika warga memiliki hambatan fungsi/disabilitas dan usia memenuhi, evaluasi ASPD.\n"
            "5. Program yang tidak cocok harus masuk program_tidak_sesuai dengan alasan spesifik.\n"
            "6. Respons hanya JSON valid. Jangan memakai markdown, heading, atau placeholder.\n"
        )
        generation_messages = [
            {"role": "system", "content": JSON_RANKING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            parsed = call_generation_api_checked(generation_messages)
        except HTTPException as e:
            logger.warning(
                "⚠️ Generation API gagal menghasilkan JSON valid, fallback deterministic dipakai: %s",
                e.detail,
            )
            parsed = build_fallback_generation(req.profil_warga, results)
        except Exception as e:
            logger.warning(
                "⚠️ Generation API Tim 1/RunPod gagal, fallback deterministic dipakai: %s",
                e,
            )
            parsed = build_fallback_generation(req.profil_warga, results)
        raise_if_parse_error(parsed)
        logger.info("DEBUG PARSED: %s", json.dumps(parsed, indent=2, ensure_ascii=False))
        parsed = enforce_program_eligibility_rules(parsed, req.profil_warga)

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


@app.post("/ask", response_model=AskResponse, tags=["RAG"],
          summary="Tanya juknis/kebijakan bantuan sosial secara bebas")
def ask(req: AskRequest):
    """
    **Mode 2: Tanya Juknis Bebas (JSON Terstruktur)**

    Contoh pertanyaan:
    - "Apa syarat penerima ASPD?"
    - "Berapa nominal bantuan PKH Plus per bulan?"
    - "Bagaimana mekanisme pencairan KIP Putri JAWARA?"

    **Response mencakup:**
    - `jawaban` — jawaban lengkap berdasarkan dokumen
    - `sumber_digunakan[]` — nama dokumen yang dirujuk
    - `poin_penting[]` — ringkasan dalam poin-poin
    - `sources[]` — metadata chunk untuk audit
    """
    check_ready(require_llm=True)
    t0 = time.time()

    try:
        top_k = req.top_k or RETRIEVAL_TOP_K
        top_n = req.top_n or RERANK_TOP_N

        # RERANKER OFF: jalur lama memakai Cross-Encoder reranking.
        # results = state.retriever.retrieve(req.query, top_k=top_k, top_n=top_n)
        results = retrieve_semantic_only(req.query, top_k=top_k, top_n=top_n)

        if not results:
            raise HTTPException(status_code=404,
                detail="Tidak ada dokumen relevan ditemukan untuk pertanyaan tersebut.")

        context = build_context_flat(results)

        final_prompt = PROMPT_TEMPLATE.format(
            system_prompt=JSON_ASK_SYSTEM_PROMPT,
            context=context,
            query=req.query,
        )

        parsed = invoke_llm(final_prompt)
        raise_if_parse_error(parsed)

        elapsed_ms = int((time.time() - t0) * 1000)

        return AskResponse(
            jawaban=parsed.get("jawaban", ""),
            sumber_digunakan=parsed.get("sumber_digunakan", []),
            poin_penting=parsed.get("poin_penting", []),
            catatan=parsed.get("catatan"),
            sources=to_source_docs(results),
            retrieval_count=len(results),
            elapsed_ms=elapsed_ms,
            model_used=RUNPOD_MODEL_NAME,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ /ask error: %s", e, exc_info=True)
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
    - Inspeksi konteks sebelum dikirim ke LLM eksternal (e.g. Gemini, GPT)
    - Debugging pipeline retrieval
    - Mengambil konteks chunk untuk diproses sendiri

    **Field `filter_programs_only`** — jika true, filter dilakukan secara
    native di Qdrant (bukan post-filter Python) via `allowed_sources`.
    """
    check_ready()
    t0 = time.time()

    try:
        top_k = req.top_k or RETRIEVAL_TOP_K
        top_n = req.top_n or RERANK_TOP_N

        # ── Step 1: Parse content → keyword query padat ─────────────────────
        effective_query = _parse_content_to_retrieval_query(req.content)
        inferred_sources = infer_retrieval_sources_from_profile(req.content)

        # ── Step 2: Gabungkan prompt retrieval sesuai program target ────────
        retrieval_prompt = retrieval_prompt_for_sources(inferred_sources)
        if retrieval_prompt and retrieval_prompt.strip():
            effective_query = f"{retrieval_prompt.strip()}\n\n{effective_query}"

        logger.info("🔎 /retrieve effective_query (200 chars): %s", effective_query[:200])

        # ── Step 3: Jalankan retrieval ──────────────────────────────────────
        # [Bug 3 Fix] filter_programs_only sekarang menggunakan native Qdrant
        # allowed_sources, bukan post-filter Python setelah retrieval.
        allowed = (
            inferred_sources
            if inferred_sources
            else list(PROGRAM_LABELS.keys()) if req.filter_programs_only else None
        )
        if inferred_sources:
            logger.info("🎯 /retrieve target program sources: %s", inferred_sources)
        # RERANKER OFF: jalur lama memakai Cross-Encoder reranking.
        # results = state.retriever.retrieve(
        #     effective_query,
        #     top_k=top_k,
        #     top_n=top_n,
        #     allowed_sources=allowed,   # [Bug 3 Fix]
        # )
        results = retrieve_semantic_only(
            effective_query,
            top_k=top_k,
            top_n=top_n,
            allowed_sources=allowed,   # [Bug 3 Fix]
        )

        if not results:
            detail = (
                "Tidak ada chunk dari 6 program utama yang relevan untuk query tersebut."
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
        port=8000,
        reload=False,
        log_level="info",
    )
