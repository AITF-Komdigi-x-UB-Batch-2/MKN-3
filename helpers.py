import re
import json
import logging
from typing import Optional

# Custom imports
from config import RETRIEVAL_TOP_K, RERANK_TOP_N
from schemas import SpesifikasiProgram, SourceDocument
from retrieval import RetrievalResult
from generation import PROGRAM_LABELS
from llm_client import parse_llm_json

logger = logging.getLogger(__name__)

# ============================================================
# RETRIEVE SYSTEM PROMPT
# ============================================================

# Query default jika program tidak terdeteksi dari profil.
RETRIEVE_SYSTEM_PROMPT = (
    "syarat kelayakan kriteria sasaran penerima manfaat "
    "besaran nominal bantuan sosial tahapan pencairan "
    "PKH Plus lanjut usia 70 tahun ASPD penyandang disabilitas "
    "juknis juklak resmi Provinsi Jawa Timur"
)

# Query khusus PKH Plus — token semantik disesuaikan dengan teks chunk kritis
# (halaman 8 juknis: "Besaran nominal bantuan sosial PKH Plus ... Rp. 2.000.000 ...
# disalurkan dalam empat tahap ... nominal masing-masing Rp. 500.000")
RETRIEVE_QUERY_PKH_PLUS = (
    "sasaran penerima lanjut usia 70 tahun ke atas desil 1 2 3 4 DTSEN "
    "besaran nominal bantuan sosial PKH Plus Rp 2000000 dua juta rupiah "
    "penyaluran empat tahap masing-masing Rp 500000 rekening Bank Jatim "
    "syarat KTP NIK KK Provinsi Jawa Timur pencairan non tunai"
)

# Query khusus ASPD — token semantik disesuaikan dengan teks chunk kritis
# (halaman 30 juklak: "Rp. 900.000 ... disalurkan dalam 4 tahapan ...")
RETRIEVE_QUERY_ASPD = (
    "sasaran penerima penyandang disabilitas usia dari 6 bulan hingga 60 tahun "
    "besaran nominal bantuan sosial ASPD Rp 900000 sembilan ratus ribu rupiah "
    "penyaluran empat tahap rekening Bank Jatim Provinsi Jawa Timur "
    "syarat kelayakan kriteria verifikasi pendamping"
)


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
    """
    parts: list[str] = []

    # ── [Bug 2 Fix] Coba safe-load sebagai JSON terlebih dahulu ──────────
    try:
        data: dict = json.loads(content)

        # Navigasi nested dict opsional (output Tim 1 bisa flat atau bersarang)
        laporan    = data.get("laporan_evaluasi") or {}
        profil_raw = laporan.get("profil_warga") or {}
        param_raw  = data.get("parameter") or {}
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

        # ── Lokasi ─────────────────────────────────────────────
        lokasi = param_raw.get("lokasi") or data.get("lokasi")
        if lokasi:
            parts.append(str(lokasi).strip()[:80])

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

        # ── Lokasi ─────────────────────────────────────────────────
        lokasi_match = re.search(
            r'(?:kec\.?|kecamatan|kelurahan|kabupaten|kota)\s+[\w\s,]+',
            content, re.IGNORECASE
        )
        if lokasi_match:
            parts.append(lokasi_match.group(0).strip()[:80])

    # ── Fallback universal: tidak ada yang terdeteksi ──────────────────
    if not parts:
        return (
            "syarat kriteria sasaran penerima bantuan sosial "
            + content[:300].strip()
        )

    # Tambahkan konteks retrieval agar semantic search lebih terarah
    base = " ".join(parts)
    return f"syarat kriteria sasaran penerima bantuan sosial: {base}"


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
        "has_disability": has_disability,
    }


def infer_retrieval_sources_from_profile(content: str) -> Optional[list[str]]:
    signals = parse_profile_signals(content)
    age = signals.get("umur")
    desil = signals.get("desil_nasional")
    has_disability = bool(signals.get("has_disability"))

    target_sources: list[str] = []

    pkh_profile_match = (
        age is not None
        and age >= 70
        and (desil is None or int(desil) <= 4)
    )
    aspd_profile_match = has_disability

    if aspd_profile_match:
        target_sources.append("Juklak ASPD Tahun 202620260225_12303533_01.pdf")

    if pkh_profile_match:
        target_sources.append("JUKNIS PKH PLUS 2026.pdf")

    return target_sources or None


# Mapping nama file sumber ke query semantik yang kaya token.
_SOURCE_SPECIFIC_QUERIES: dict[str, str] = {
    "JUKNIS PKH PLUS 2026.pdf": RETRIEVE_QUERY_PKH_PLUS,
    "Juklak ASPD Tahun 202620260225_12303533_01.pdf": RETRIEVE_QUERY_ASPD,
}


def retrieval_prompt_for_sources(sources: Optional[list[str]]) -> str:
    """
    Buat query retrieval semantik berdasarkan daftar program yang terdeteksi.

    Jika hanya 1 program terdeteksi → gunakan query spesifik program tersebut
    agar embedding lebih terarah ke chunk nominal + mekanisme pencairan.
    Jika lebih dari 1 atau tidak ada → gunakan gabungan query atau fallback default.
    """
    if not sources:
        return RETRIEVE_SYSTEM_PROMPT

    # Kasus 1 program — pakai query semantik spesifik
    if len(sources) == 1:
        specific = _SOURCE_SPECIFIC_QUERIES.get(sources[0])
        if specific:
            return specific

    # Kasus multi-program — gabungkan query spesifik per program
    parts = [_SOURCE_SPECIFIC_QUERIES.get(src) for src in sources]
    parts = [p for p in parts if p]  # filter None
    if parts:
        return " ".join(parts)

    # Fallback ke default jika tidak ada query spesifik terdaftar
    return RETRIEVE_SYSTEM_PROMPT


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
        for key in ["parameter", "kesimpulan"]:
            if key not in data and key in laporan:
                data[key] = laporan[key]
    return data


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
