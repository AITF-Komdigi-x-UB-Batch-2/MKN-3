# ============================================================
# webservice.py — FastAPI REST API untuk SIRA RAG System
# Social Welfare Policy Recommender System (Tim 3 — MKN3)
#
# Endpoints:
#   POST /recommend   → Profil warga → ranking program bantuan (JSON terstruktur)
#   POST /ask         → Tanya juknis bebas (JSON terstruktur)
#   GET  /health      → Status Qdrant + model
#   GET  /programs    → Daftar 6 program bantuan
#
# Jalankan:
#   uvicorn webservice:app --host 0.0.0.0 --port 8000 --reload
#
# Swagger UI:
#   http://localhost:8000/docs
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
    EMBED_MODEL_NAME, RERANKER_MODEL_NAME,
    OLLAMA_BASE_URL, OLLAMA_GENERATION_MODEL, OLLAMA_TEMPERATURE,
    PROMPT_TEMPLATE, POLICY_PROMPT_TEMPLATE,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
    configure_utf8_stdio,
)
configure_utf8_stdio()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_ollama import OllamaLLM
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

JSON_RANKING_SYSTEM_PROMPT = """Anda adalah SIRA (Sistem Rekomender Intervensi & Kebijakan Program Sosial), \
asisten pakar kebijakan sosial milik Tim 3 Universitas Brawijaya.

Tugas Anda: Berdasarkan PROFIL WARGA dan KONTEKS PROGRAM BANTUAN yang disediakan, \
evaluasi kelayakan warga HANYA untuk 6 program utama berikut:
1. Asistensi Sosial Penyandang Disabilitas (ASPD)
2. Penanganan Kemiskinan Ekstrem
3. PKH Plus (Lanjut Usia 70+)
4. KIP KPM JAWARA (Kewirausahaan KPM)
5. KIP PPKS JAWARA (Penyandang Masalah Sosial)
6. KIP Putri JAWARA (Perempuan Tangguh)

=== INSTRUKSI PENTING ===
1. Evaluasi hanya 6 program utama di atas secara individual.
2. Tentukan status: "ELIGIBLE", "MUNGKIN_ELIGIBLE", atau "TIDAK_ELIGIBLE".
3. Ranking dari yang paling cocok ke yang paling tidak cocok.
4. Sertakan spesifikasi teknis: nominal bantuan, frekuensi, syarat dokumen, mekanisme.
5. Berikan reasoning yang jelas dan WAJIB mengutip sumber dokumen.
6. JANGAN merekomendasikan program di luar 6 program utama.
7. DILARANG menyebut Program Sembako, PKH reguler, BPNT, PBI Jaminan Kesehatan, Rutilahu, PIP, Jamkesda, atau bantuan tambahan lain.

=== FORMAT OUTPUT ===
Anda WAJIB merespons HANYA dengan JSON valid berikut, tanpa teks apapun di luar JSON:

{
  "ringkasan_profil": "rangkuman singkat kondisi kunci warga yang relevan",
  "rekomendasi": [
    {
      "rank": 1,
      "nama_program": "Nama Program",
      "status": "ELIGIBLE",
      "dasar_hukum": "Nama dokumen dan bagian yang relevan",
      "alasan_kelayakan": "Penjelasan mengapa warga layak berdasarkan kriteria dokumen",
      "spesifikasi": {
        "nominal_bantuan": "Rp X.XXX.XXX / periode",
        "frekuensi": "per bulan / per tahun / dst",
        "sasaran": "kriteria penerima sesuai juknis",
        "syarat_dokumen": ["KTP", "KK", "dst"],
        "mekanisme": "cara pencairan/penyaluran"
      }
    }
  ],
  "program_tidak_sesuai": [
    {
      "nama_program": "Nama Program",
      "status": "TIDAK_ELIGIBLE",
      "alasan": "kondisi warga yang tidak memenuhi kriteria"
    }
  ]
}
"""

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
# SINGLETON STATE
# ============================================================

class AppState:
    retriever: Optional[PolicyRetriever] = None
    llm: Optional[OllamaLLM] = None
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
        state.llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_GENERATION_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            num_ctx=8192,
        )
        state.ready = True
        state.startup_time = round(time.time() - t0, 2)
        logger.info("✅ Semua model siap dalam %.2fs.", state.startup_time)
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
        "Pipeline: `Qdrant Semantic Search` → `Cross-Encoder Reranking` → `Ollama LLM Generation`\n\n"
        "Tim 3 — Universitas Brawijaya × DISKOMINFO Jawa Timur (AITF 2026)"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC — Request Models
# ============================================================

class RecommendRequest(BaseModel):
    profil_warga: str = Field(
        ...,
        min_length=10,
        description="Deskripsi profil warga (teks bebas atau JSON string)",
        examples=["Warga lanjut usia 72 tahun, tinggal sendiri, tidak punya penghasilan tetap, rumah tidak layak huni, belum terdaftar DTKS"]
    )
    scoring_result: Optional[str] = Field(
        default="",
        description="Hasil scoring MKN1 (opsional). Sertakan desil dan skor dari sistem MKN1."
    )
    top_k: Optional[int] = Field(default=None, ge=5, le=100, description=f"Kandidat awal semantic search (default: {RETRIEVAL_TOP_K})")
    top_n: Optional[int] = Field(default=None, ge=1, le=20, description=f"Finalis setelah reranking (default: {RERANK_TOP_N})")


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        description="Pertanyaan terkait juknis/kebijakan bantuan sosial",
        examples=["Apa syarat penerima PKH Plus untuk lanjut usia?"]
    )
    top_k: Optional[int] = Field(default=None, ge=5, le=100)
    top_n: Optional[int] = Field(default=None, ge=1, le=20)


# ============================================================
# PYDANTIC — Request Model MKN1 Integration
# Menerima output langsung dari Tim 1 (model scoring/evaluasi)
# ============================================================

class DisabilitasDetail(BaseModel):
    penglihatan: Optional[str] = None
    pendengaran: Optional[str] = None
    berjalan_naik_tangga: Optional[str] = None
    menggunakan_tangan_jari: Optional[str] = None
    belajar_intelektual: Optional[str] = None
    pengendalian_perilaku: Optional[str] = None
    berbicara_komunikasi: Optional[str] = None
    mengurus_diri: Optional[str] = None
    mengingat_berkonsentrasi: Optional[str] = None
    kesedihan_depresi: Optional[str] = None


class UsahaDetail(BaseModel):
    jenis_usaha: Optional[list[str]] = []
    omset_usaha_utama: Optional[str] = None


class ProfilWargaMKN1(BaseModel):
    nik: Optional[str] = None
    nama: Optional[str] = None
    umur: Optional[int] = None
    hubungan_kepala_keluarga: Optional[str] = None
    status_perkawinan: Optional[str] = None
    jumlah_anggota_keluarga: Optional[int] = None


class AnalisisMKN1(BaseModel):
    demografi: Optional[str] = None
    ekonomi: Optional[str] = None
    infrastruktur_hunian: Optional[str] = None
    kesehatan_gizi: Optional[str] = None
    disabilitas_fungsi: Optional[str] = None
    sintesis_pkh_plus: Optional[str] = None
    sintesis_aspd: Optional[str] = None


class LaporanEvaluasiMKN1(BaseModel):
    profil_warga: Optional[ProfilWargaMKN1] = None
    analisis: Optional[AnalisisMKN1] = None


class KesimpulanProgram(BaseModel):
    status_kelayakan: Optional[str] = None
    urgensi: Optional[str] = None
    label: Optional[int] = None


class KesimpulanMKN1(BaseModel):
    pkh_plus: Optional[KesimpulanProgram] = None
    aspd: Optional[KesimpulanProgram] = None


class ParameterMKN1(BaseModel):
    desil_nasional: Optional[int] = None
    penguasaan_bangunan: Optional[str] = None
    luas_bangunan_m2: Optional[float] = None
    kondisi_gizi: Optional[str] = None
    penyakit_menahun: Optional[str] = None
    disabilitas: Optional[DisabilitasDetail] = None
    status_dtsekolah: Optional[str] = None
    lokasi: Optional[str] = None
    usaha: Optional[UsahaDetail] = None
    izin_usaha: Optional[str] = None


class SkorMKN1(BaseModel):
    skor_pkh_plus: Optional[float] = None
    skor_aspd: Optional[float] = None


class MKN1Request(BaseModel):
    """Model request yang menerima output langsung dari Tim 1 (MKN1)."""
    laporan_evaluasi: Optional[LaporanEvaluasiMKN1] = None
    parameter: Optional[ParameterMKN1] = None
    skor: Optional[SkorMKN1] = None
    kesimpulan: Optional[KesimpulanMKN1] = None
    top_k: Optional[int] = Field(default=None, ge=5, le=100, description=f"Kandidat awal semantic search (default: {RETRIEVAL_TOP_K})")
    top_n: Optional[int] = Field(default=None, ge=1, le=20, description=f"Finalis setelah reranking (default: {RERANK_TOP_N})")


# ============================================================
# PYDANTIC — Response Models (terstruktur untuk tim MVP)
# ============================================================

class SpesifikasiProgram(BaseModel):
    nominal_bantuan: Optional[str] = None
    frekuensi: Optional[str] = None
    sasaran: Optional[str] = None
    syarat_dokumen: Optional[list[str]] = None
    mekanisme: Optional[str] = None


class RekomendasiProgram(BaseModel):
    rank: int
    nama_program: str
    status: str   # ELIGIBLE | MUNGKIN_ELIGIBLE | TIDAK_ELIGIBLE
    dasar_hukum: Optional[str] = None
    alasan_kelayakan: Optional[str] = None
    spesifikasi: Optional[SpesifikasiProgram] = None


class ProgramTidakSesuai(BaseModel):
    nama_program: str
    status: str
    alasan: Optional[str] = None


class SourceDocument(BaseModel):
    program: str
    sumber: str
    judul_halaman: Optional[str] = None
    page_number: Optional[str] = None
    rerank_score: float
    embed_score: float
    text_preview: str


class RecommendResponse(BaseModel):
    ringkasan_profil: str
    rekomendasi: list[RekomendasiProgram]
    program_tidak_sesuai: list[ProgramTidakSesuai]
    # Metadata pipeline
    sources: list[SourceDocument]
    retrieval_count: int
    program_count: int
    elapsed_ms: int
    model_used: str


class AskResponse(BaseModel):
    jawaban: str
    sumber_digunakan: list[str]
    poin_penting: list[str]
    catatan: Optional[str] = None
    # Metadata pipeline
    sources: list[SourceDocument]
    retrieval_count: int
    elapsed_ms: int
    model_used: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
    startup_time_s: Optional[float] = None
    startup_error: Optional[str] = None
    config: dict


class ProgramInfo(BaseModel):
    filename: str
    nama_program: str


# ============================================================
# HELPERS
# ============================================================

def check_ready():
    if not state.ready or state.retriever is None or state.llm is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service belum siap. Model sedang loading atau gagal inisialisasi.",
                "startup_error": state.startup_error,
            }
        )


def to_source_docs(results: list[RetrievalResult]) -> list[SourceDocument]:
    return [
        SourceDocument(
            program=PROGRAM_LABELS.get(
                r.metadata.get("sumber", ""),
                r.metadata.get("sumber", "").replace(".pdf", "")
            ),
            sumber=r.metadata.get("sumber", "unknown"),
            judul_halaman=r.metadata.get("judul_halaman"),
            page_number=str(r.metadata.get("page_number", "")),
            rerank_score=round(r.score, 4),
            embed_score=round(r.embed_score, 4),
            text_preview=r.text[:200].strip(),
        )
        for r in results
    ]


def parse_llm_json(raw: str) -> dict:
    """
    Ekstrak dan parse JSON dari output LLM.
    Handle kasus LLM membungkus JSON dengan ```json``` atau teks tambahan.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Cari blok JSON pertama
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("⚠️ LLM tidak menghasilkan JSON valid. Fallback ke raw.")
    return {"_raw": raw, "_parse_error": True}


def invoke_llm(prompt: str) -> dict:
    raw = state.llm.invoke(prompt)
    if not isinstance(raw, str):
        raw = raw.content
    return parse_llm_json(raw)


def raise_if_parse_error(parsed: dict):
    if parsed.get("_parse_error"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "LLM tidak menghasilkan JSON valid. Coba ulangi request.",
                "raw_output_preview": parsed.get("_raw", "")[:500],
            }
        )


# ============================================================
# PARSER MKN1 → Teks Profil Warga
# ============================================================

def parse_mkn1_to_profil_warga(req: MKN1Request) -> tuple[str, str]:
    """
    Mengonversi output terstruktur Tim 1 (MKN1) menjadi dua string:
    - profil_warga : teks deskriptif untuk semantic search & LLM context
    - scoring_result: ringkasan skor & kesimpulan MKN1 untuk prompt LLM

    Returns:
        (profil_warga: str, scoring_result: str)
    """
    parts: list[str] = []

    # ── Profil Dasar ──────────────────────────────────────────
    profil = req.laporan_evaluasi.profil_warga if req.laporan_evaluasi else None
    if profil:
        lines = []
        if profil.nama:
            lines.append(f"Nama: {profil.nama}")
        if profil.umur:
            lines.append(f"Umur: {profil.umur} tahun")
        if profil.status_perkawinan:
            lines.append(f"Status perkawinan: {profil.status_perkawinan}")
        if profil.jumlah_anggota_keluarga:
            lines.append(f"Jumlah anggota keluarga: {profil.jumlah_anggota_keluarga} orang")
        if profil.hubungan_kepala_keluarga:
            lines.append(f"Hubungan dengan kepala keluarga: {profil.hubungan_kepala_keluarga}")
        if lines:
            parts.append("=== PROFIL WARGA ===\n" + "\n".join(lines))

    # ── Parameter Ekonomi & Lokasi ────────────────────────────
    param = req.parameter
    if param:
        eco_lines = []
        if param.desil_nasional is not None:
            eco_lines.append(f"Desil kemiskinan nasional: {param.desil_nasional} (DTSEN)")
        if param.status_dtsekolah:
            eco_lines.append(f"Status DTSEN/DTKS: {param.status_dtsekolah}")
        if param.lokasi:
            eco_lines.append(f"Domisili: {param.lokasi}")
        if param.penguasaan_bangunan:
            eco_lines.append(f"Status penguasaan bangunan: {param.penguasaan_bangunan}")
        if param.luas_bangunan_m2:
            eco_lines.append(f"Luas bangunan: {param.luas_bangunan_m2} m\u00b2")
        if param.kondisi_gizi:
            eco_lines.append(f"Kondisi gizi: {param.kondisi_gizi}")
        if param.penyakit_menahun:
            eco_lines.append(f"Penyakit menahun: {param.penyakit_menahun}")
        if eco_lines:
            parts.append("=== DATA EKONOMI & LOKASI ===\n" + "\n".join(eco_lines))

        # ── Disabilitas ────────────────────────────────────────
        disabilitas = param.disabilitas
        if disabilitas:
            KESULITAN_KEYWORDS = ["banyak kesulitan", "beberapa kesulitan", "tidak bisa"]
            aktif = {
                k: v for k, v in disabilitas.model_dump().items()
                if v and any(kw in v.lower() for kw in KESULITAN_KEYWORDS)
            }
            if aktif:
                dis_lines = [f"- {k.replace('_', ' ').title()}: {v}" for k, v in aktif.items()]
                parts.append("=== KONDISI DISABILITAS ===\nWarga mengalami hambatan pada:\n" + "\n".join(dis_lines))
            else:
                parts.append("=== KONDISI DISABILITAS ===\nTidak ada hambatan fungsi yang signifikan.")

        # ── Usaha ──────────────────────────────────────────────
        usaha = param.usaha
        if usaha:
            usaha_lines = []
            if usaha.jenis_usaha:
                usaha_lines.append(f"Jenis usaha: {', '.join(usaha.jenis_usaha)}")
            else:
                usaha_lines.append("Tidak memiliki usaha aktif.")
            if usaha.omset_usaha_utama:
                usaha_lines.append(f"Omset usaha utama: {usaha.omset_usaha_utama}")
            parts.append("=== DATA USAHA ===\n" + "\n".join(usaha_lines))

    # ── Analisis Naratif MKN1 ──────────────────────────────────
    analisis = req.laporan_evaluasi.analisis if req.laporan_evaluasi else None
    if analisis:
        analisis_lines = []
        for field_name in ["demografi", "ekonomi", "infrastruktur_hunian",
                           "kesehatan_gizi", "disabilitas_fungsi"]:
            val = getattr(analisis, field_name, None)
            if val:
                analisis_lines.append(f"- {field_name.replace('_', ' ').title()}: {val}")
        if analisis_lines:
            parts.append("=== ANALISIS MKN1 ===\n" + "\n".join(analisis_lines))

    profil_warga_text = "\n\n".join(parts)

    # ── Scoring & Kesimpulan (untuk scoring_result) ────────────
    scoring_parts: list[str] = []

    skor = req.skor
    if skor:
        scoring_parts.append("Skor Kelayakan dari Model MKN1:")
        if skor.skor_pkh_plus is not None:
            scoring_parts.append(f"  - PKH Plus : {skor.skor_pkh_plus:.4f}")
        if skor.skor_aspd is not None:
            scoring_parts.append(f"  - ASPD     : {skor.skor_aspd:.4f}")

    kesimpulan = req.kesimpulan
    if kesimpulan:
        scoring_parts.append("\nKesimpulan Model MKN1:")
        if kesimpulan.pkh_plus:
            k = kesimpulan.pkh_plus
            scoring_parts.append(
                f"  - PKH Plus : {k.status_kelayakan} "
                f"(Urgensi: {k.urgensi}, Label: {k.label})"
            )
        if kesimpulan.aspd:
            k = kesimpulan.aspd
            scoring_parts.append(
                f"  - ASPD     : {k.status_kelayakan} "
                f"(Urgensi: {k.urgensi}, Label: {k.label})"
            )

    analisis = req.laporan_evaluasi.analisis if req.laporan_evaluasi else None
    if analisis:
        if analisis.sintesis_pkh_plus:
            scoring_parts.append(f"\nSintesis PKH Plus (MKN1): {analisis.sintesis_pkh_plus}")
        if analisis.sintesis_aspd:
            scoring_parts.append(f"Sintesis ASPD (MKN1): {analisis.sintesis_aspd}")

    scoring_result_text = "\n".join(scoring_parts)

    return profil_warga_text, scoring_result_text


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
            "reranker": RERANKER_MODEL_NAME,
            "llm_model": OLLAMA_GENERATION_MODEL,
            "default_top_k": RETRIEVAL_TOP_K,
            "default_top_n": RERANK_TOP_N,
        }
    )


@app.get("/programs", response_model=list[ProgramInfo], tags=["System"],
         summary="Daftar program bantuan sosial dalam knowledge base")
def list_programs():
    """Kembalikan daftar 6 program bantuan sosial yang tersedia."""
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

    Field `scoring_result` opsional: isi dengan output MKN1 (skor 0-100, desil 1-10)
    untuk rekomendasi yang lebih presisi.

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

        query_for_retrieval = (
            f"kriteria sasaran penerima bantuan sosial syarat dokumen nominal "
            f"berdasarkan profil: {req.profil_warga[:300]}"
        )

        results = state.retriever.retrieve(query_for_retrieval, top_k=top_k, top_n=top_n)
        results = [
            r for r in results
            if r.metadata.get("sumber", "") in PROGRAM_LABELS
        ]

        if not results:
            raise HTTPException(status_code=404,
                detail="Tidak ada dokumen 6 program utama yang relevan ditemukan untuk profil yang diberikan.")

        context = build_context_grouped(results)

        profil_section = f"=== PROFIL WARGA ===\n{req.profil_warga}"
        if req.scoring_result:
            profil_section += f"\n\n=== HASIL SCORING MKN1 ===\n{req.scoring_result}"

        if req.scoring_result:
            final_prompt = POLICY_PROMPT_TEMPLATE.format(
                system_prompt=JSON_RANKING_SYSTEM_PROMPT,
                scoring_result=profil_section,
                context=context,
            )
        else:
            final_prompt = PROMPT_TEMPLATE.format(
                system_prompt=JSON_RANKING_SYSTEM_PROMPT,
                context=context,
                query=profil_section,
            )

        parsed = invoke_llm(final_prompt)
        raise_if_parse_error(parsed)

        elapsed_ms = int((time.time() - t0) * 1000)
        program_count = len(set(r.metadata.get("sumber", "") for r in results))

        rekomendasi = [
            RekomendasiProgram(
                rank=item.get("rank", i + 1),
                nama_program=item.get("nama_program", ""),
                status=item.get("status", ""),
                dasar_hukum=item.get("dasar_hukum"),
                alasan_kelayakan=item.get("alasan_kelayakan"),
                spesifikasi=SpesifikasiProgram(**item["spesifikasi"]) if item.get("spesifikasi") else None,
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

        return RecommendResponse(
            ringkasan_profil=parsed.get("ringkasan_profil", ""),
            rekomendasi=rekomendasi,
            program_tidak_sesuai=tidak_sesuai,
            sources=to_source_docs(results),
            retrieval_count=len(results),
            program_count=program_count,
            elapsed_ms=elapsed_ms,
            model_used=OLLAMA_GENERATION_MODEL,
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
    check_ready()
    t0 = time.time()

    try:
        top_k = req.top_k or RETRIEVAL_TOP_K
        top_n = req.top_n or RERANK_TOP_N

        results = state.retriever.retrieve(req.query, top_k=top_k, top_n=top_n)

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
            model_used=OLLAMA_GENERATION_MODEL,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ /ask error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/recommend-from-mkn1", response_model=RecommendResponse, tags=["RAG"],
          summary="Rekomendasi langsung dari output JSON Tim 1 (MKN1)")
def recommend_from_mkn1(req: MKN1Request):
    """
    **Mode Integrasi Tim 1 → Tim 3: JSON MKN1 → Rekomendasi Program Bantuan**

    Endpoint ini menerima **output JSON langsung dari model Tim 1 (MKN1)**
    tanpa perlu konversi manual. Parser akan otomatis mengubahnya menjadi
    teks profil dan skor untuk sistem RAG Tim 3.

    **Alur:**
    `Output JSON MKN1` → `Parser MKN1` → `Profil Teks + Skor` → `RAG Pipeline` → `Ranking 6 Program`

    **Contoh penggunaan:**
    Tempelkan langsung JSON dari endpoint MKN1 ke dalam Body request ini.
    """
    check_ready()
    t0 = time.time()

    try:
        # ── Parse output MKN1 ke format RAG ────────────────────
        profil_warga, scoring_result = parse_mkn1_to_profil_warga(req)

        if len(profil_warga.strip()) < 20:
            raise HTTPException(
                status_code=422,
                detail="Data dari MKN1 tidak cukup untuk membuat profil warga. "
                       "Pastikan field laporan_evaluasi dan parameter terisi."
            )

        logger.info("📥 MKN1 parsed:\n%s", profil_warga[:300])

        top_k = req.top_k or RETRIEVAL_TOP_K
        top_n = req.top_n or RERANK_TOP_N

        query_for_retrieval = (
            f"kriteria sasaran penerima bantuan sosial syarat dokumen nominal "
            f"berdasarkan profil: {profil_warga[:300]}"
        )

        results = state.retriever.retrieve(query_for_retrieval, top_k=top_k, top_n=top_n)
        results = [
            r for r in results
            if r.metadata.get("sumber", "") in PROGRAM_LABELS
        ]

        if not results:
            raise HTTPException(
                status_code=404,
                detail="Tidak ada dokumen 6 program utama yang relevan ditemukan."
            )

        context = build_context_grouped(results)

        profil_section = f"=== PROFIL WARGA ===\n{profil_warga}"
        if scoring_result:
            profil_section += f"\n\n=== HASIL SCORING MKN1 ===\n{scoring_result}"

        if scoring_result:
            final_prompt = POLICY_PROMPT_TEMPLATE.format(
                system_prompt=JSON_RANKING_SYSTEM_PROMPT,
                scoring_result=profil_section,
                context=context,
            )
        else:
            final_prompt = PROMPT_TEMPLATE.format(
                system_prompt=JSON_RANKING_SYSTEM_PROMPT,
                context=context,
                query=profil_section,
            )

        parsed = invoke_llm(final_prompt)
        raise_if_parse_error(parsed)

        elapsed_ms = int((time.time() - t0) * 1000)
        program_count = len(set(r.metadata.get("sumber", "") for r in results))

        rekomendasi = [
            RekomendasiProgram(
                rank=item.get("rank", i + 1),
                nama_program=item.get("nama_program", ""),
                status=item.get("status", ""),
                dasar_hukum=item.get("dasar_hukum"),
                alasan_kelayakan=item.get("alasan_kelayakan"),
                spesifikasi=SpesifikasiProgram(**item["spesifikasi"]) if item.get("spesifikasi") else None,
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

        return RecommendResponse(
            ringkasan_profil=parsed.get("ringkasan_profil", ""),
            rekomendasi=rekomendasi,
            program_tidak_sesuai=tidak_sesuai,
            sources=to_source_docs(results),
            retrieval_count=len(results),
            program_count=program_count,
            elapsed_ms=elapsed_ms,
            model_used=OLLAMA_GENERATION_MODEL,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ /recommend-from-mkn1 error: %s", e, exc_info=True)
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
