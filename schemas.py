import json
from typing import Optional
from pydantic import BaseModel, Field, model_validator

# Import defaults and models from config and generation
from config import RETRIEVAL_TOP_K, RERANK_TOP_N, RUNPOD_MODEL_NAME

# ============================================================
# PYDANTIC — Request Models
# ============================================================

class RecommendRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "profil_warga": "- NIK / No. KK     : PRS_d042e02a62905683082f73a210ec7c037ec1268f9442876ea48a264fda186bea / FAM_f4d5cdb8947bb34051be28581ec19375a613e0d5aae9525e433e9b764c858acb\n- Nama             : ******YAH\n- Umur             : 12 tahun\n- Hub. Kepala KK   : Anak\n- Status Kawin     : Belum kawin\n- Jml. Anggota KK  : 4 orang\n- Desil Nasional   : 1 | Status DTSEN: DTSEN AKTIF\n- Status Keberadaan: Ditemukan / Aktif\n- Bansos           : PKH, SEMBAKO\n- PBI Jaminan Kes  : Ya\n- Kondisi Gizi     : Tidak diketahui\n- Penyakit Menahun : Gagal ginjal\nHambatan Fungsi:\n- Penglihatan      : Tidak mengalami kesulitan | Pendengaran: Ya, banyak kesulitan dan membutuhkan bantuan\n- Berjalan/Tangga  : Ya, banyak kesulitan dan membutuhkan bantuan | Tangan/Jari: Ya, banyak kesulitan dan membutuhkan bantuan\n- Belajar/Intelek  : Ya, banyak kesulitan dan membutuhkan bantuan | Perilaku: Ya, banyak kesulitan dan membutuhkan bantuan\n- Bicara/Komunikasi: Ya, banyak kesulitan dan membutuhkan bantuan | Mengurus Diri: Ya, banyak kesulitan dan membutuhkan bantuan\n- Ingatan/Fokus    : Ya, banyak kesulitan dan membutuhkan bantuan | Sedih/Depresi: Ya, banyak kesulitan dan membutuhkan bantuan\n- Wilayah          : Polowijen, Kec. Blimbing, Kota Malang, Jawa Timur",
                "top_k": 5,
            }
        }
    }

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
                user_content = str(msg.get("content") or msg.get("query") or "").strip()

        if user_content:
            self.profil_warga = user_content
            return self

        raise ValueError(
            "Isi salah satu: `profil_warga`, `content`, atau `messages` dengan message role `user` yang memiliki content."
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
    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "- NIK / No. KK     : PRS_d042e02a62905683082f73a210ec7c037ec1268f9442876ea48a264fda186bea / FAM_f4d5cdb8947bb34051be28581ec19375a613e0d5aae9525e433e9b764c858acb\n- Nama             : ******YAH\n- Umur             : 12 tahun\n- Hub. Kepala KK   : Anak\n- Status Kawin     : Belum kawin\n- Jml. Anggota KK  : 4 orang\n- Desil Nasional   : 1 | Status DTSEN: DTSEN AKTIF\n- Status Keberadaan: Ditemukan / Aktif\n- Bansos           : PKH, SEMBAKO\n- PBI Jaminan Kes  : Ya\n- Kondisi Gizi     : Tidak diketahui\n- Penyakit Menahun : Gagal ginjal\nHambatan Fungsi:\n- Penglihatan      : Tidak mengalami kesulitan | Pendengaran: Ya, banyak kesulitan dan membutuhkan bantuan\n- Berjalan/Tangga  : Ya, banyak kesulitan dan membutuhkan bantuan | Tangan/Jari: Ya, banyak kesulitan dan membutuhkan bantuan\n- Belajar/Intelek  : Ya, banyak kesulitan dan membutuhkan bantuan | Perilaku: Ya, banyak kesulitan dan membutuhkan bantuan\n- Bicara/Komunikasi: Ya, banyak kesulitan dan membutuhkan bantuan | Mengurus Diri: Ya, banyak kesulitan dan membutuhkan bantuan\n- Ingatan/Fokus    : Ya, banyak kesulitan dan membutuhkan bantuan | Sedih/Depresi: Ya, banyak kesulitan dan membutuhkan bantuan\n- Wilayah          : Polowijen, Kec. Blimbing, Kota Malang, Jawa Timur",
                "filter_programs_only": True,
                "top_k": 5,
            }
        }
    }

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
                user_content = str(msg.get("content") or msg.get("query") or "").strip()
                if user_content:
                    break

        if user_content:
            self.content = user_content
            return self

        raise ValueError(
            "Field `content` wajib diisi, atau kirim `messages` yang memiliki message role `user` berisi content."
        )


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
    sumber: Optional[str] = None
    alasan_kelayakan: Optional[str] = None


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
    rekomendasi_teknis_bansos: Optional[str] = None
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
