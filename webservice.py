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

JSON_RANKING_SYSTEM_PROMPT = """Anda adalah AI Auditor resmi Dinas Sosial Provinsi Jawa Timur yang bertugas melakukan verifikasi dan validasi kelayakan penerima manfaat dua program bantuan sosial.
    
    Tugas Anda: Berdasarkan PROFIL WARGA dan KONTEKS PROGRAM BANTUAN yang disediakan, \
    evaluasi kelayakan warga HANYA untuk 2 program utama berikut:
    1. Asistensi Sosial Penyandang Disabilitas (ASPD)
    2. PKH Plus (Lanjut Usia 70+)
    
    === INSTRUKSI PENTING ===
    1. Evaluasi hanya 2 program utama di atas secara individual.
    2. Tentukan status: "ELIGIBLE", atau "TIDAK_ELIGIBLE".
    3. Ranking dari yang paling cocok ke yang paling tidak cocok.
    4. Sertakan spesifikasi teknis: nominal bantuan, frekuensi, syarat dokumen, mekanisme.
    5. Berikan reasoning yang jelas dan WAJIB mengutip sumber dokumen.
    6. JANGAN merekomendasikan program di luar 2 program utama.
    7. DILARANG menyebut Program Sembako, PKH reguler, BPNT, PBI Jaminan Kesehatan, Rutilahu, PIP, Jamkesda, atau bantuan tambahan lain.
    8. Hard rule PKH Plus: hanya untuk lanjut usia 70 tahun ke atas. Jika umur warga kurang dari 70 tahun, PKH Plus WAJIB TIDAK_ELIGIBLE walaupun desil/DTSEN aktif atau ada hambatan fungsi.
    9. Hambatan fungsi/disabilitas dievaluasi untuk ASPD, bukan alasan meloloskan PKH Plus.
    
    === FORMAT OUTPUT ===
    Anda WAJIB merespons HANYA dengan JSON valid tanpa markdown dan tanpa teks pembuka/penutup.
    Gunakan key berikut persis:
    - ringkasan_profil: string konkret berisi umur, desil, DTSEN/DTKS, disabilitas/usia lansia, dan kondisi kunci warga.
    - rekomendasi: array program yang ELIGIBLE. Setiap item wajib berisi rank, nama_program, status, dasar_hukum, alasan_kelayakan, spesifikasi.
    - spesifikasi: object berisi nominal_bantuan, frekuensi, sasaran, syarat_dokumen, mekanisme.
    - program_tidak_sesuai: array program TIDAK_ELIGIBLE. Setiap item wajib berisi nama_program, status, alasan.
    
    Larangan keras:
    - Jangan menyalin placeholder seperti "Nama Program", "Rp X.XXX.XXX", "dst", "rangkuman singkat", atau "Penjelasan mengapa".
    - Jangan mengosongkan alasan. Semua alasan harus merujuk kondisi warga dan kriteria dokumen.
    - nama_program harus salah satu dari 2 program utama yang disebut di atas.
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
        state.ready = True
        state.startup_time = round(time.time() - t0, 2)
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
# PYDANTIC — Request Models
# ============================================================

class RecommendRequest(BaseModel):
    profil_warga: Optional[str] = Field(
        default=None,
        min_length=10,
        description="Deskripsi profil warga (teks bebas atau JSON string)",
        examples=["Warga lanjut usia 72 tahun, tinggal sendiri, tidak punya penghasilan tetap, rumah tidak layak huni, belum terdaftar DTKS"]
    )
    content: Optional[str] = Field(
        default=None,
        min_length=10,
        description="Alias untuk profil_warga. Cocok untuk payload query dari Tim 4."
    )
    messages: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Format chat JSONL dari Tim 4. Jika profil_warga/content kosong, "
            "message role user dipakai sebagai profil."
        )
    )
    scoring_result: Optional[str] = Field(
        default="",
        description="Hasil scoring MKN1 (opsional). Sertakan desil dan skor dari sistem MKN1."
    )
    top_k: Optional[int] = Field(default=None, ge=5, le=100, description=f"Kandidat awal semantic search (default: {RETRIEVAL_TOP_K})")
    top_n: Optional[int] = Field(default=None, ge=1, le=20, description=f"Finalis semantic search (default: {RERANK_TOP_N})")

    @model_validator(mode="after")
    def normalize_profile_from_tim4_payload(self):
        if self.profil_warga and self.profil_warga.strip():
            self.profil_warga = self.profil_warga.strip()
            return self

        if self.content and self.content.strip():
            self.profil_warga = self.content.strip()
            return self

        user_content = ""
        for msg in self.messages or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                user_content = str(msg.get("content") or "").strip()

        if user_content:
            self.profil_warga = user_content
            return self

        raise ValueError(
            "Isi salah satu: `profil_warga`, `content`, atau `messages` dengan message role `user`."
        )


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        description="Pertanyaan terkait juknis/kebijakan bantuan sosial",
        examples=["Apa syarat penerima PKH Plus untuk lanjut usia?"]
    )
    top_k: Optional[int] = Field(default=None, ge=5, le=100)
    top_n: Optional[int] = Field(default=None, ge=1, le=20)


class RetrieveOnlyRequest(BaseModel):
    """
    Request model untuk endpoint /retrieve.

    Kirim teks profil warga (free-text / key-value) di field `content`, atau
    kirim satu baris JSONL fine-tuning di field `messages`.
    Sistem akan otomatis mengekstrak keyword penting dan menggabungkan
    `RETRIEVE_SYSTEM_PROMPT` (yang bisa diubah di kode) sebagai prefix query.
    """
    content: Optional[str] = Field(
        default=None,
        min_length=10,
        description=(
            "Teks profil warga lengkap (free-text / key-value). "
            "Sistem otomatis mengekstrak keyword usia, desil, disabilitas, DTKS/DTSEN, "
            "skor prioritas, dan lokasi menjadi query retrieval yang padat."
        ),
        examples=[
            "Profil Warga:\n- Umur: 76 tahun\n- Desil Nasional: 2\n- Status DTSEN: DTSEN AKTIF\n"
            "Skor PKH Plus: 0.7045 (prioritas tinggi)\nSkor ASPD: 0.0 (prioritas rendah)"
        ]
    )
    messages: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Opsional. Format chat JSONL seperti sample_retrieval_bansos_final.jsonl. "
            "Jika `content` kosong, endpoint akan mengambil `content` dari message role `user`."
        ),
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100,
        description=f"Kandidat dari Qdrant semantic search (default: {RETRIEVAL_TOP_K})"
    )
    top_n: Optional[int] = Field(
        default=None, ge=1, le=50,
        description=f"Finalis semantic search (default: {RERANK_TOP_N})"
    )
    filter_programs_only: bool = Field(
        default=False,
        description="Jika true, hanya tampilkan chunk dari 6 program utama (filter berdasarkan field 'sumber')"
    )

    @model_validator(mode="after")
    def normalize_content_from_messages(self):
        if self.content and self.content.strip():
            self.content = self.content.strip()
            return self

        user_content = ""
        for msg in self.messages or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                user_content = str(msg.get("content") or "").strip()
                if user_content:
                    break

        if user_content:
            self.content = user_content
            return self

        raise ValueError(
            "Field `content` wajib diisi, atau kirim `messages` yang memiliki message role `user` berisi content."
        )


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
    top_n: Optional[int] = Field(default=None, ge=1, le=20, description=f"Finalis semantic search (default: {RERANK_TOP_N})")


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
    rerank_score: float  # Legacy field; semantic-only mode mengisi nilai ini dari embed_score.
    embed_score: float
    text_preview: str


class RetrieveChunkResult(BaseModel):
    rank: int
    program: str
    sumber: str
    judul_halaman: Optional[str] = None
    page_number: Optional[str] = None
    rerank_score: float  # Legacy field; semantic-only mode mengisi nilai ini dari embed_score.
    embed_score: float
    text: str           # Teks lengkap chunk (bukan preview)
    metadata: dict      # Seluruh metadata payload dari Qdrant


class RetrieveOnlyResponse(BaseModel):
    retrieval_count: int
    programs_covered: int
    elapsed_ms: int
    results: list[RetrieveChunkResult]


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

# ============================================================
# PARSER — Konversi teks profil warga panjang → query retrieval
# ============================================================

def _parse_content_to_retrieval_query(content: str) -> str:
    """
    Ekstrak informasi kunci dari teks profil warga panjang (free-text / key-value
    / JSON string mentah dari Tim 1) menjadi query retrieval yang padat dan
    bermakna semantik untuk BGE-M3.

    [Bug 2 Fix] Tahap pertama: coba parse `content` sebagai JSON.
    Jika berhasil, ekstrak nilai kunci secara langsung dari dict/nested dict
    (mendukung format output Tim 1 yang mengandung field seperti `umur`,
    `desil_nasional`, `disabilitas`, dll. — baik di root maupun di bawah
    `laporan_evaluasi.profil_warga` atau `parameter`).
    Jika gagal di-parse (bukan JSON), fallback ke logika Regex yang sudah ada.

    Informasi yang diekstrak (kedua jalur):
    - Usia (angka + tahun)
    - Desil kemiskinan nasional
    - Status DTKS / DTSEN
    - Kondisi disabilitas (jika ada hambatan fungsi)
    - Jenis kelamin / perempuan
    - Skor program prioritas (PKH Plus, ASPD, dll.)
    - Lokasi domisili (kecamatan/kabupaten)
    - Kepemilikan usaha
    """
    parts: list[str] = []

    # ── [Bug 2 Fix] Coba safe-load sebagai JSON terlebih dahulu ──────────
    try:
        data: dict = json.loads(content)

        # Navigasi nested dict opsional (output Tim 1 bisa flat atau bersarang)
        laporan    = data.get("laporan_evaluasi") or {}
        profil_raw = laporan.get("profil_warga") or {}
        param_raw  = data.get("parameter") or {}
        skor_raw   = data.get("skor") or {}
        analisis   = laporan.get("analisis") or {}

        # ── Usia ──────────────────────────────────────────────
        umur = profil_raw.get("umur") or data.get("umur")
        if umur is not None:
            try:
                age = int(umur)
                parts.append(f"usia {age} tahun")
                if age >= 70:
                    parts.append("lanjut usia 70 tahun ke atas")
            except (ValueError, TypeError):
                pass

        # ── Desil ──────────────────────────────────────────────
        desil = param_raw.get("desil_nasional") or data.get("desil_nasional")
        if desil is not None:
            parts.append(f"desil {desil}")

        # ── Status DTKS / DTSEN ────────────────────────────────
        status_dt = (
            param_raw.get("status_dtsekolah")
            or data.get("status_dtsekolah")
            or ""
        ).lower()
        if any(k in status_dt for k in ["dtsen aktif", "dtks aktif", "aktif"]):
            parts.append("terdaftar DTSEN DTKS")
        elif any(k in status_dt for k in ["dtsen", "dtks"]):
            parts.append("DTSEN DTKS")

        # ── Disabilitas ────────────────────────────────────────
        # Dua sumber: field disabilitas dari parameter, atau analisis.disabilitas_fungsi
        KESULITAN_KW = ["banyak kesulitan", "beberapa kesulitan", "tidak bisa", "tidak mampu"]
        dis_obj = param_raw.get("disabilitas") or data.get("disabilitas") or {}
        has_disability = any(
            v and any(kw in str(v).lower() for kw in KESULITAN_KW)
            for v in dis_obj.values()
        ) if isinstance(dis_obj, dict) else False
        # Fallback ke analisis naratif
        if not has_disability:
            dis_narasi = str(analisis.get("disabilitas_fungsi") or "").lower()
            has_disability = any(kw in dis_narasi for kw in KESULITAN_KW)
        if has_disability:
            parts.append("penyandang disabilitas")

        # ── Jenis Kelamin ──────────────────────────────────────
        nama = str(profil_raw.get("nama") or data.get("nama") or "").lower()
        hub  = str(profil_raw.get("hubungan_kepala_keluarga") or "").lower()
        if re.search(r'\bperempuan\b|\bistri\b|\bibu\b', f"{nama} {hub}"):
            parts.append("perempuan")

        # ── Skor Prioritas Program ─────────────────────────────
        skor_pkh  = skor_raw.get("skor_pkh_plus")
        skor_aspd = skor_raw.get("skor_aspd")
        eligible_programs: list[str] = []
        if skor_pkh is not None and float(skor_pkh) > 0.3:
            eligible_programs.append("PKH Plus")
        if skor_aspd is not None and float(skor_aspd) > 0.3:
            eligible_programs.append("ASPD")
        if eligible_programs:
            parts.append(f"prioritas program: {', '.join(eligible_programs)}")

        # ── Lokasi ─────────────────────────────────────────────
        lokasi = param_raw.get("lokasi") or data.get("lokasi")
        if lokasi:
            parts.append(str(lokasi).strip()[:80])

        # ── Usaha / Wirausaha ──────────────────────────────────
        usaha_obj = param_raw.get("usaha") or data.get("usaha") or {}
        jenis_usaha = usaha_obj.get("jenis_usaha") if isinstance(usaha_obj, dict) else None
        if jenis_usaha:
            parts.append("memiliki usaha")

        logger.info("🔑 JSON parse berhasil: %d komponen query diekstrak.", len(parts))

    except (json.JSONDecodeError, TypeError, AttributeError):
        # ── [Bug 2 Fallback] Bukan JSON → jalankan logika Regex lama ─────
        logger.debug("ℹ️ JSON parse gagal, fallback ke Regex parser.")

        # ── Usia ──────────────────────────────────────────────────
        age_match = re.search(r'(\d+)\s*tahun', content, re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
            parts.append(f"usia {age} tahun")
            if age >= 70:
                parts.append("lanjut usia 70 tahun ke atas")

        # ── Desil ──────────────────────────────────────────────────
        desil_match = re.search(r'desil\s*(?:nasional)?\s*[:\-]?\s*(\d+)', content, re.IGNORECASE)
        if desil_match:
            parts.append(f"desil {desil_match.group(1)}")

        # ── Status DTKS / DTSEN ────────────────────────────────────
        c_lower = content.lower()
        if any(k in c_lower for k in ['dtsen aktif', 'dtks aktif', 'terdaftar dtks', 'terdaftar dtsen']):
            parts.append("terdaftar DTSEN DTKS")
        elif any(k in c_lower for k in ['dtsen', 'dtks']):
            parts.append("DTSEN DTKS")

        # ── Disabilitas ────────────────────────────────────────────
        KESULITAN_KW_RE = ["banyak kesulitan", "beberapa kesulitan", "tidak bisa", "tidak mampu"]
        has_disability = any(kw in content.lower() for kw in KESULITAN_KW_RE)
        if has_disability:
            parts.append("penyandang disabilitas")

        # ── Jenis Kelamin ──────────────────────────────────────────
        if re.search(r'\bperempuan\b|\bistri\b|\bibu\b', content, re.IGNORECASE):
            parts.append("perempuan")

        # ── Skor Prioritas Program ─────────────────────────────────
        # Format: "Skor PKH Plus    : 0.7045 (prioritas tinggi)"
        skor_matches = re.findall(
            r'skor\s+([\w\s+]+?)\s*[:\-]\s*([0-9.]+)',
            content, re.IGNORECASE
        )
        eligible_programs: list[str] = []
        for program_name, skor_str in skor_matches:
            try:
                skor = float(skor_str)
                if skor > 0.3:
                    eligible_programs.append(program_name.strip())
            except ValueError:
                pass
        if eligible_programs:
            parts.append(f"prioritas program: {', '.join(eligible_programs)}")

        # ── Lokasi ─────────────────────────────────────────────────
        lokasi_match = re.search(
            r'(?:kec\.?|kecamatan|kelurahan|kabupaten|kota)\s+[\w\s,]+',
            content, re.IGNORECASE
        )
        if lokasi_match:
            parts.append(lokasi_match.group(0).strip()[:80])

        # ── Usaha / Wirausaha ──────────────────────────────────────
        USAHA_KW = ["jenis usaha", "wirausaha", "berdagang", "omset usaha"]
        if any(kw in content.lower() for kw in USAHA_KW):
            parts.append("memiliki usaha")

    # ── Fallback universal: tidak ada yang terdeteksi ──────────────────
    if not parts:
        return (
            "syarat kriteria sasaran penerima bantuan sosial "
            + content[:300].strip()
        )

    # Tambahkan konteks retrieval agar semantic search lebih terarah
    base = " ".join(parts)
    return f"syarat kriteria sasaran penerima bantuan sosial: {base}"


def check_ready(require_llm: bool = False):
    if not state.ready or state.retriever is None or (require_llm and state.llm is None):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service belum siap. Model sedang loading atau gagal inisialisasi.",
                "startup_error": state.startup_error,
            }
        )


def normalize_spesifikasi(raw: object) -> Optional[SpesifikasiProgram]:
    if not isinstance(raw, dict):
        return None

    data = raw.copy()
    syarat = data.get("syarat_dokumen")

    if isinstance(syarat, str):
        data["syarat_dokumen"] = [
            item.strip()
            for item in re.split(r"[,;\n]+", syarat)
            if item.strip()
        ]
    elif syarat is None:
        data["syarat_dokumen"] = None
    elif not isinstance(syarat, list):
        data["syarat_dokumen"] = [str(syarat)]
    else:
        data["syarat_dokumen"] = [
            str(item).strip()
            for item in syarat
            if str(item).strip()
        ]

    for key in ["nominal_bantuan", "frekuensi", "sasaran", "mekanisme"]:
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            data[key] = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value)
            )

    return SpesifikasiProgram(**data)


def normalize_tim1_output(raw: str) -> dict:
    if not raw or not raw.strip():
        return {}

    parsed = parse_llm_json(raw)
    if parsed.get("_parse_error"):
        return {}

    data = parsed.copy()
    laporan = data.get("laporan_evaluasi")
    if isinstance(laporan, dict):
        for key in ["parameter", "skor", "kesimpulan"]:
            if key not in data and key in laporan:
                data[key] = laporan[key]
    return data


def tim1_is_layak(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    status = str(item.get("status_kelayakan") or "").upper()
    label = item.get("label")
    return ("LAYAK" in status and "TIDAK" not in status) or label == 1


def parse_profile_signals(profil_warga: str) -> dict:
    text = profil_warga or ""
    lower = text.lower()

    def number(pattern: str, cast=float):
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        try:
            return cast(match.group(1))
        except (TypeError, ValueError):
            return None

    age = number(r"umur\s*[:\-]?\s*(\d+)", int)
    if age is None:
        age = number(r"(\d+)\s*tahun", int)

    desil = number(r"desil\s*(?:nasional)?\s*[:\-]?\s*(\d+)", int)
    skor_pkh = number(r"skor\s+pkh\s*plus\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", float)
    if skor_pkh is None:
        skor_pkh = number(r"pkh\+\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", float)
    skor_aspd = number(r"skor\s+aspd\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", float)
    if skor_aspd is None:
        skor_aspd = number(r"aspd\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", float)

    status_dtsen = None
    status_match = re.search(r"status\s+dtsen\s*[:\-]?\s*([^\n]+)", text, re.IGNORECASE)
    if status_match:
        status_dtsen = status_match.group(1).strip()

    lokasi = None
    lokasi_match = re.search(r"wilayah\s*[:\-]?\s*([^\n]+)", text, re.IGNORECASE)
    if lokasi_match:
        lokasi = lokasi_match.group(1).strip()

    no_disability = all(kw in lower for kw in [
        "berjalan/tangga  : tidak mengalami kesulitan".lower(),
        "mengurus diri    : tidak mengalami kesulitan".lower(),
    ])
    has_disability = any(kw in lower for kw in [
        "banyak kesulitan",
        "tidak bisa",
        "membutuhkan bantuan",
        "disabilitas",
        "bedridden",
        "bed ridden",
    ]) and not no_disability

    return {
        "umur": age,
        "desil_nasional": desil,
        "status_dtsen": status_dtsen,
        "lokasi": lokasi,
        "skor_pkh_plus": skor_pkh,
        "skor_aspd": skor_aspd,
        "has_disability": has_disability,
    }


def infer_retrieval_sources_from_profile(content: str) -> Optional[list[str]]:
    signals = parse_profile_signals(content)
    age = signals.get("umur")
    desil = signals.get("desil_nasional")
    skor_pkh = signals.get("skor_pkh_plus")
    skor_aspd = signals.get("skor_aspd")
    has_disability = bool(signals.get("has_disability"))

    target_sources: list[str] = []

    pkh_score_positive = skor_pkh is not None and float(skor_pkh) > 0.3
    aspd_score_positive = skor_aspd is not None and float(skor_aspd) > 0.3

    pkh_profile_match = (
        age is not None
        and age >= 70
        and (desil is None or int(desil) <= 4)
        and (skor_pkh is None or float(skor_pkh) > 0.05)
    )
    aspd_profile_match = has_disability and (skor_aspd is None or float(skor_aspd) > 0.05)

    if aspd_score_positive or aspd_profile_match:
        target_sources.append("Juklak ASPD Tahun 202620260225_12303533_01.pdf")

    if pkh_score_positive or pkh_profile_match:
        target_sources.append("JUKNIS PKH PLUS 2026.pdf")

    return target_sources or None


def retrieval_prompt_for_sources(sources: Optional[list[str]]) -> str:
    if not sources:
        return RETRIEVE_SYSTEM_PROMPT

    program_names = [
        PROGRAM_LABELS.get(source, source.replace(".pdf", ""))
        for source in sources
    ]
    return (
        "Temukan syarat kelayakan, kriteria sasaran penerima, besaran nominal bantuan, "
        "dan mekanisme pencairan untuk program berikut berdasarkan petunjuk teknis resmi: "
        + "; ".join(program_names)
        + "."
    )


def normalize_program_name(name: str) -> str:
    lower = (name or "").lower()
    if "pkh" in lower and "plus" in lower:
        return "PKH Plus (Lanjut Usia 70+)"
    if "aspd" in lower or "disabilitas" in lower:
        return "Asistensi Sosial Penyandang Disabilitas (ASPD)"
    for program_name in PROGRAM_LABELS.values():
        if lower == program_name.lower():
            return program_name
    return name or ""


def enforce_program_eligibility_rules(parsed: dict, profil_warga: str) -> dict:
    """
    Guardrail deterministik setelah LLM.
    LLM tidak boleh meloloskan program yang melanggar hard rule juknis.
    """
    if not isinstance(parsed, dict):
        return parsed

    signals = parse_profile_signals(profil_warga)
    age = signals.get("umur")
    skor_pkh = signals.get("skor_pkh_plus")

    data = parsed.copy()
    rekomendasi_raw = data.get("rekomendasi") if isinstance(data.get("rekomendasi"), list) else []
    tidak_sesuai_raw = (
        data.get("program_tidak_sesuai")
        if isinstance(data.get("program_tidak_sesuai"), list)
        else []
    )

    rekomendasi: list[dict] = []
    tidak_sesuai: list[dict] = [
        item.copy() for item in tidak_sesuai_raw if isinstance(item, dict)
    ]

    def add_tidak_sesuai(program_name: str, alasan: str):
        canonical = normalize_program_name(program_name)
        for item in tidak_sesuai:
            if normalize_program_name(str(item.get("nama_program") or "")) == canonical:
                item["nama_program"] = canonical
                item["status"] = "TIDAK_ELIGIBLE"
                item["alasan"] = alasan
                return
        tidak_sesuai.append({
            "nama_program": canonical,
            "status": "TIDAK_ELIGIBLE",
            "alasan": alasan,
        })

    for item in rekomendasi_raw:
        if not isinstance(item, dict):
            continue
        current = item.copy()
        canonical = normalize_program_name(str(current.get("nama_program") or ""))
        current["nama_program"] = canonical

        is_pkh_plus = canonical == "PKH Plus (Lanjut Usia 70+)"
        if is_pkh_plus and age is not None and age < 70:
            alasan = (
                f"Tidak memenuhi hard rule PKH Plus: usia warga {age} tahun, "
                "sedangkan sasaran PKH Plus adalah lanjut usia 70 tahun ke atas."
            )
            if skor_pkh is not None:
                alasan += f" Skor PKH Plus dari profil: {skor_pkh}."
            add_tidak_sesuai(canonical, alasan)
            continue

        if is_pkh_plus and skor_pkh is not None and float(skor_pkh) <= 0.05:
            add_tidak_sesuai(
                canonical,
                f"Tidak direkomendasikan karena skor PKH Plus dari profil adalah {skor_pkh}, "
                "di bawah ambang prioritas."
            )
            continue

        rekomendasi.append(current)

    for idx, item in enumerate(rekomendasi, 1):
        item["rank"] = idx

    data["rekomendasi"] = rekomendasi
    data["program_tidak_sesuai"] = tidak_sesuai
    return data


def source_ref_for_program(results: list[RetrievalResult], source_filename: str) -> str:
    pages = [
        str(r.metadata.get("page_number", ""))
        for r in results
        if r.metadata.get("sumber") == source_filename and r.metadata.get("page_number") not in (None, "")
    ]
    unique_pages = []
    for page in pages:
        if page not in unique_pages:
            unique_pages.append(page)
    if unique_pages:
        return f"{source_filename}, Hal. {', '.join(unique_pages[:3])}"
    return source_filename


def build_fallback_generation(
    profil_warga: str,
    scoring_result: str,
    results: list[RetrievalResult],
) -> dict:
    tim1 = normalize_tim1_output(scoring_result)
    laporan = tim1.get("laporan_evaluasi") if isinstance(tim1.get("laporan_evaluasi"), dict) else {}
    profil = laporan.get("profil_warga") if isinstance(laporan.get("profil_warga"), dict) else {}
    analisis = laporan.get("analisis") if isinstance(laporan.get("analisis"), dict) else {}
    parameter = tim1.get("parameter") if isinstance(tim1.get("parameter"), dict) else {}
    kesimpulan = tim1.get("kesimpulan") if isinstance(tim1.get("kesimpulan"), dict) else {}
    skor = tim1.get("skor") if isinstance(tim1.get("skor"), dict) else {}
    profile_signals = parse_profile_signals(profil_warga)

    umur = profil.get("umur") or profile_signals.get("umur")
    desil = parameter.get("desil_nasional") or profile_signals.get("desil_nasional")
    status_dtsen = (
        profil.get("status_dtsen")
        or parameter.get("status_dtsekolah")
        or profile_signals.get("status_dtsen")
    )
    wilayah = profil.get("wilayah")
    if isinstance(wilayah, dict):
        wilayah_text = ", ".join(str(v) for v in wilayah.values() if v)
    else:
        wilayah_text = str(wilayah or profile_signals.get("lokasi") or "")

    ringkasan_parts = []
    if umur is not None:
        ringkasan_parts.append(f"umur {umur} tahun")
    if desil is not None:
        ringkasan_parts.append(f"desil nasional {desil}")
    if status_dtsen:
        ringkasan_parts.append(str(status_dtsen))
    if wilayah_text:
        ringkasan_parts.append(wilayah_text)
    if analisis.get("disabilitas_fungsi"):
        ringkasan_parts.append(str(analisis["disabilitas_fungsi"]))
    ringkasan = (
        "Profil warga: " + "; ".join(ringkasan_parts)
        if ringkasan_parts
        else profil_warga[:500]
    )

    program_configs = [
        (
            "pkh_plus",
            "PKH Plus (Lanjut Usia 70+)",
            "JUKNIS PKH PLUS 2026.pdf",
            analisis.get("sintesis_pkh_plus"),
            skor.get("skor_pkh_plus", profile_signals.get("skor_pkh_plus")),
            {
                "nominal_bantuan": "Mengacu JUKNIS PKH Plus 2026",
                "frekuensi": "sesuai tahapan penyaluran dalam juknis",
                "sasaran": "lansia 70 tahun ke atas yang memenuhi kriteria DTSEN/desil dan administrasi kependudukan Jawa Timur",
                "syarat_dokumen": ["KTP", "KK", "NIK"],
                "mekanisme": "verifikasi/pemutakhiran data dan penyaluran sesuai petunjuk teknis PKH Plus",
            },
        ),
        (
            "aspd",
            "Asistensi Sosial Penyandang Disabilitas (ASPD)",
            "Juklak ASPD Tahun 202620260225_12303533_01.pdf",
            analisis.get("sintesis_aspd"),
            skor.get("skor_aspd", profile_signals.get("skor_aspd")),
            {
                "nominal_bantuan": "Mengacu Juklak ASPD Tahun 2026",
                "frekuensi": "sesuai tahapan penyaluran dalam juklak",
                "sasaran": "penyandang disabilitas yang memenuhi kriteria usia, domisili, desil/prioritas, dan verifikasi lapangan",
                "syarat_dokumen": ["KTP", "KK", "NIK", "dokumen pendukung disabilitas/verifikasi"],
                "mekanisme": "verifikasi data penerima, penetapan, dan penyaluran melalui mekanisme juklak ASPD",
            },
        ),
    ]

    rekomendasi = []
    tidak_sesuai = []
    rank = 1
    for key, program_name, source, sintesis, score, spec in program_configs:
        kes = kesimpulan.get(key)
        inferred_layak = False
        if key == "pkh_plus":
            inferred_layak = (
                umur is not None and umur >= 70
                and desil is not None and desil <= 4
                and status_dtsen and "aktif" in str(status_dtsen).lower()
            )
            if not sintesis and inferred_layak:
                sintesis = (
                    f"Warga berusia {umur} tahun, memenuhi batas lansia 70 tahun ke atas; "
                    f"desil nasional {desil} masuk prioritas 1-4; status DTSEN aktif."
                )
        elif key == "aspd":
            inferred_layak = (
                profile_signals.get("has_disability")
                and umur is not None and umur <= 60
                and desil is not None and desil <= 5
            )
            if not sintesis and inferred_layak:
                sintesis = (
                    f"Warga memiliki indikasi hambatan fungsi/disabilitas, usia {umur} tahun "
                    f"masuk rentang ASPD, dan desil nasional {desil} masuk prioritas."
                )

        alasan = str(sintesis or "Tidak ada sintesis Tim 1 yang tersedia.")
        if score is not None:
            alasan = f"{alasan} Skor Tim 1: {score}."

        if tim1_is_layak(kes) or inferred_layak:
            rekomendasi.append({
                "rank": rank,
                "nama_program": program_name,
                "status": "ELIGIBLE",
                "dasar_hukum": source_ref_for_program(results, source),
                "alasan_kelayakan": alasan,
                "spesifikasi": spec,
            })
            rank += 1
        else:
            tidak_sesuai.append({
                "nama_program": program_name,
                "status": "TIDAK_ELIGIBLE",
                "alasan": alasan,
            })

    for program_name in PROGRAM_LABELS.values():
        if program_name not in [r["nama_program"] for r in rekomendasi] and program_name not in [r["nama_program"] for r in tidak_sesuai]:
            tidak_sesuai.append({
                "nama_program": program_name,
                "status": "TIDAK_ELIGIBLE",
                "alasan": "Tidak ada indikator profil dan hasil Tim 1 yang menunjukkan kecocokan utama untuk program ini.",
            })

    return {
        "ringkasan_profil": ringkasan,
        "rekomendasi": rekomendasi,
        "program_tidak_sesuai": tidak_sesuai,
    }


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
    if state.llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM lokal tidak aktif. Gunakan endpoint /recommend yang memakai API RunPod.",
        )
    raw = state.llm.invoke(prompt)
    if not isinstance(raw, str):
        raw = raw.content
    return parse_llm_json(raw)


def extract_chat_content(api_response: dict) -> str:
    try:
        content = api_response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
    except (KeyError, IndexError, TypeError):
        pass

    raise ValueError("Response API generation tidak memiliki choices[0].message.content.")


def call_runpod_chat_api(api_url: str, messages: list[dict]) -> str:
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY belum terisi di .env.")

    payload = {
        "model": RUNPOD_MODEL_NAME,
        "messages": messages,
        "temperature": RUNPOD_TEMPERATURE,
        "max_tokens": RUNPOD_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=TIM1_API_TIMEOUT_S) as client:
            resp = client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            return extract_chat_content(resp.json())
    except httpx.HTTPStatusError as e:
        response = e.response
        body_preview = response.text[:1000] if response is not None else ""
        raise RuntimeError(
            f"Gagal call API Tim 1/RunPod ({api_url}): "
            f"HTTP {response.status_code if response is not None else 'unknown'} "
            f"{response.reason_phrase if response is not None else ''}. "
            f"Response body: {body_preview}"
        ) from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Gagal call API Tim 1/RunPod ({api_url}): {e}") from e


def call_classification_api(profil_warga: str) -> str:
    return call_runpod_chat_api(
        TIM1_CLASSIFICATION_API_URL,
        [{"role": "user", "content": profil_warga}],
    )


def call_generation_api(messages: list[dict]) -> dict:
    raw_content = call_runpod_chat_api(TIM1_GENERATION_API_URL, messages)
    return parse_llm_json(raw_content)


def is_placeholder_generation(parsed: dict) -> bool:
    if not isinstance(parsed, dict):
        return True

    placeholder_terms = [
        "rangkuman singkat",
        "nama program",
        "rp x.xxx.xxx",
        "penjelasan mengapa",
        "nama dokumen dan bagian",
        "kriteria penerima sesuai juknis",
        "cara pencairan/penyaluran",
    ]

    def contains_placeholder(value) -> bool:
        if isinstance(value, str):
            lower = value.lower()
            return any(term in lower for term in placeholder_terms)
        if isinstance(value, dict):
            return any(contains_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(contains_placeholder(v) for v in value)
        return False

    if contains_placeholder(parsed):
        return True

    rekomendasi = parsed.get("rekomendasi")
    tidak_sesuai = parsed.get("program_tidak_sesuai")
    if not isinstance(rekomendasi, list) or not isinstance(tidak_sesuai, list):
        return True

    return False


def call_generation_api_checked(messages: list[dict]) -> dict:
    parsed = call_generation_api(messages)
    if not is_placeholder_generation(parsed):
        return parsed

    retry_messages = messages + [
        {
            "role": "user",
            "content": (
                "Output sebelumnya ditolak karena mengulang prompt atau menyalin placeholder schema. "
                "Jawab ulang HANYA dengan JSON final. Isi semua field dengan nilai konkret berdasarkan "
                "profil warga, hasil Tim 1, dan konteks dokumen. Jangan menulis ulang instruksi, "
                "jangan memakai markdown, dan jangan memakai placeholder."
            ),
        }
    ]
    parsed_retry = call_generation_api(retry_messages)
    if is_placeholder_generation(parsed_retry):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Model generation masih menyalin placeholder schema setelah retry.",
                "output_preview": json.dumps(parsed_retry, ensure_ascii=False)[:700],
            },
        )
    return parsed_retry


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
            eco_lines.append(f"Luas bangunan: {param.luas_bangunan_m2} m²")
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
            # "reranker": RERANKER_MODEL_NAME,  # Dinonaktifkan untuk uji coba semantic-only.
            "llm_model": RUNPOD_MODEL_NAME,
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
        # [Bug 3 Fix] top_k dinaikkan ke minimal 40 agar semua 6 program
        # terwakili dalam pool semantic search, terutama dokumen ASPD dan KIP JAWARA
        # yang kadang kalah saing dengan PKH Plus di embedding score.
        top_k = max(req.top_k or RETRIEVAL_TOP_K, 40)
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

        scoring_result = req.scoring_result or ""
        if not scoring_result:
            try:
                scoring_result = call_classification_api(req.profil_warga)
                logger.info("✅ Klasifikasi Tim 1 diterima (%d chars).", len(scoring_result))
            except Exception as e:
                logger.warning("⚠️ Klasifikasi Tim 1 gagal, lanjut tanpa scoring_result: %s", e)

        profil_section = f"=== PROFIL WARGA ===\n{req.profil_warga}"
        if scoring_result:
            profil_section += f"\n\n=== HASIL SCORING MKN1 ===\n{scoring_result}"

        user_prompt = (
            "=== PROFIL WARGA DARI TIM 4 (ACUAN UTAMA) ===\n"
            f"{req.profil_warga}\n"
            "=== AKHIR PROFIL WARGA ===\n\n"
            "=== HASIL KLASIFIKASI / SCORING TIM 1 ===\n"
            f"{scoring_result or 'Tidak tersedia. Gunakan profil warga dan konteks dokumen.'}\n"
            "=== AKHIR HASIL TIM 1 ===\n\n"
            "=== KONTEKS DOKUMEN KEBIJAKAN DARI RETRIEVAL ===\n"
            f"{context}\n"
            "=== AKHIR KONTEKS DOKUMEN ===\n\n"
            "INSTRUKSI EKSEKUSI:\n"
            "1. Isi JSON dengan data konkret dari profil warga, hasil Tim 1, dan konteks dokumen.\n"
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
            parsed = build_fallback_generation(req.profil_warga, scoring_result, results)
        except Exception as e:
            logger.warning(
                "⚠️ Generation API Tim 1/RunPod gagal, fallback deterministic dipakai: %s",
                e,
            )
            parsed = build_fallback_generation(req.profil_warga, scoring_result, results)
        raise_if_parse_error(parsed)
        parsed = enforce_program_eligibility_rules(parsed, req.profil_warga)

        elapsed_ms = int((time.time() - t0) * 1000)
        program_count = len(set(r.metadata.get("sumber", "") for r in results))

        rekomendasi = [
            RekomendasiProgram(
                rank=item.get("rank", i + 1),
                nama_program=item.get("nama_program", ""),
                status=item.get("status", ""),
                dasar_hukum=item.get("dasar_hukum"),
                alasan_kelayakan=item.get("alasan_kelayakan"),
                spesifikasi=normalize_spesifikasi(item.get("spesifikasi")),
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
    
