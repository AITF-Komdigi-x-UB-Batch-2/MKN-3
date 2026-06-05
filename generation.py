# ============================================================
# generation.py — Stage F: RAG Generation (MKN3 — Final)
# Social Welfare Policy Recommender System (Tim 4)
#
# Pipeline: Profil Warga → Multi-Query Retrieve (per program) → LLM Ranking
#
# Arsitektur:
#   - Retrieval  : retrieval.py (multi-query + filter Qdrant per dokumen)
#   - Generation : API model eksternal via webservice.py
#   - Prompt     : RANKING_SYSTEM_PROMPT dari config.py
# ============================================================

import json
import logging
import re
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from config import (
    POLICY_PROMPT_TEMPLATE, PROMPT_TEMPLATE, RANKING_SYSTEM_PROMPT,
    QDRANT_COLLECTION, EMBED_MODEL_NAME, RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
    LLM_PROVIDER, HF_GENERATION_MODEL, HF_TOKEN,
    RUNPOD_MODEL_NAME,
    configure_utf8_stdio,
)
configure_utf8_stdio()
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box

from retrieval import PolicyRetriever, RetrievalResult

# ============================================================
# MAPPING: file PDF → nama program bantuan
# ============================================================
PROGRAM_LABELS = {
    "Juklak ASPD Tahun 202620260225_12303533_01.pdf":
        "Asistensi Sosial Penyandang Disabilitas (ASPD)",
    "JUKNIS KEMISKINAN EKSTREM (13-1-2025)-1 (1) (2).pdf":
        "Penanganan Kemiskinan Ekstrem",
    "JUKNIS PKH PLUS 2026.pdf":
        "PKH Plus (Lanjut Usia 70+)",
    "PETUNJUK TEKNIS KIP KPM JAWARA.pdf":
        "KIP KPM JAWARA (Kewirausahaan KPM)",
    "Petunjuk Teknis KIP PPKS Jawara 2026.pdf":
        "KIP PPKS JAWARA (Penyandang Masalah Sosial)",
    "PETUNJUK TEKNIS KIP PUTRI JAWARA.pdf":
        "KIP Putri JAWARA (Perempuan Tangguh)",
}

# Filter retrieval per program — cegah chunk antar program tercampur
PROGRAM_FILE_MAP = {
    "ASPD":      ["Juklak ASPD Tahun 202620260225_12303533_01.pdf"],
    "Ekstrem":   ["JUKNIS KEMISKINAN EKSTREM (13-1-2025)-1 (1) (2).pdf"],
    "PKH Plus":  ["JUKNIS PKH PLUS 2026.pdf"],
    "KIP Jawara": [
        "PETUNJUK TEKNIS KIP KPM JAWARA.pdf",
        "Petunjuk Teknis KIP PPKS Jawara 2026.pdf",
        "PETUNJUK TEKNIS KIP PUTRI JAWARA.pdf",
    ],
}

PROGRAM_DISPLAY_NAMES = {
    "ASPD": "Asistensi Sosial Penyandang Disabilitas (ASPD)",
    "Kemiskinan Ekstrem": "Penanganan Kemiskinan Ekstrem",
    "PKH Plus": "PKH Plus (Lanjut Usia 70+)",
    "KIP KPM Jawara": "KIP KPM JAWARA (Kewirausahaan KPM)",
    "KIP PPKS Jawara": "KIP PPKS JAWARA (Penyandang Masalah Sosial)",
    "KIP Putri Jawara": "KIP Putri JAWARA (Perempuan Tangguh)",
}

NOMINAL_SOURCE_PATH = Path(__file__).resolve().parent / "chunked_data" / "juknis_extracted_normalized.jsonl"

# ============================================================
# PROFIL PARSER — Structural highlighting untuk LLM
# ============================================================

def parse_profil(profil_warga: str) -> str:
    """
    Tambahkan ringkasan terstruktur di atas teks profil asli.
    Mencegah LLM salah baca data kunci (desil, DTKS, kondisi khusus).
    """
    p_lower = profil_warga.lower()

    desil_m = re.search(r'desil\s*(\d+)', profil_warga, re.IGNORECASE)
    desil = f"Desil {desil_m.group(1)}" if desil_m else "Tidak disebutkan"

    dtks = "Ya" if any(k in p_lower for k in [
        'dtks', 'dtsen', 'terdata dalam', 'terdaftar dalam'
    ]) else "Tidak disebutkan"

    disabilitas = "Ya" if 'disabilitas' in p_lower else "Tidak"

    ages = [int(a) for a in re.findall(r'(\d+)\s*tahun', p_lower)]
    is_lansia_kw = any(kw in p_lower for kw in ["lansia", "lanjut usia", "kakek", "nenek"])

    lansia_70  = "Ya" if any(a >= 70 for a in ages) or is_lansia_kw else "Tidak"
    perempuan  = "Ya" if re.search(r'(perempuan|istri|ibu)', profil_warga, re.IGNORECASE) else "Tidak"

    # Deteksi potensi usaha — termasuk keterampilan implisit (jahit, dll.)
    usaha_kw = ["usaha", "wirausaha", "berdagang", "jualan", "dagang",
                "jahit", "bengkel", "warung", "toko", "kue", "masak"]
    usaha = "Ya" if any(kw in p_lower for kw in usaha_kw) else "Tidak"

    jml_m  = re.search(r'(\d+)\s*orang', profil_warga, re.IGNORECASE)
    jumlah = jml_m.group(1) if jml_m else "Tidak disebutkan"

    lokasi_m = re.search(
        r'(kecamatan|kelurahan|kota|kabupaten)\s+[\w\s]+',
        profil_warga, re.IGNORECASE
    )
    lokasi = lokasi_m.group(0).strip()[:60] if lokasi_m else "Tidak disebutkan"

    return f"""=== POIN KUNCI PROFIL (ACUAN UTAMA — JANGAN DIUBAH ATAU DIASUMSIKAN) ===
⚠️  Gunakan data ini sebagai acuan utama. JANGAN timpa dengan data dari dokumen.

  Desil DTSEN/kemiskinan     : {desil}
  Terdaftar DTKS/DTSEN       : {dtks}
  Lokasi domisili            : {lokasi}
  Ada anggota disabilitas    : {disabilitas}
  Ada anggota lansia 70+     : {lansia_70}
  Ada anggota perempuan      : {perempuan}
  Ada potensi/aktivitas usaha: {usaha}
  Jumlah anggota keluarga    : {jumlah} orang

CATATAN PENCOCOKAN SYARAT:
- Cocokkan angka desil dan usia secara ketat dengan syarat dokumen.
- Terdaftar DTKS/DTSEN tidak otomatis memenuhi syarat jika dokumen meminta desil tertentu.
- Jika profil eksplisit di luar angka/rentang syarat dokumen, status program harus TIDAK ELIGIBLE.
- Jika satu syarat wajib TIDAK MEMENUHI, jangan jadikan program MUNGKIN ELIGIBLE hanya karena syarat lain cocok.

=== TEKS PROFIL LENGKAP (baca seluruhnya sebelum menilai) ===
{profil_warga}"""


# ============================================================
# CONTEXT BUILDER - Group per 6 Program Utama
# ============================================================

def build_context_grouped(results: list[RetrievalResult]) -> str:
    if not results:
        return "(Tidak ada dokumen relevan yang ditemukan.)"

    juknis_groups: dict[str, list[RetrievalResult]] = {}

    for r in results:
        src = r.metadata.get("sumber", "unknown")
        if src in PROGRAM_LABELS:
            juknis_groups.setdefault(src, []).append(r)

    parts = []

    # ── Juknis 6 Program Utama ──
    for prog_idx, (src, program_name) in enumerate(PROGRAM_LABELS.items(), 1):
        chunks = juknis_groups.get(src, [])
        section = f"=== PROGRAM {prog_idx}: {program_name} ===\n(Sumber: {src})\n\n"
        chunk_texts = []
        for r in chunks:
            judul   = r.metadata.get("judul_halaman", "")
            halaman = r.metadata.get("page_number", "?")
            loc     = f"[Hal. {halaman}{' | ' + judul if judul else ''}]"
            chunk_texts.append(f"{loc}\n{r.text.strip()}")
        if chunk_texts:
            section += "\n---\n".join(chunk_texts)
        else:
            section += "(Tidak ada chunk relevan yang ditemukan untuk program ini.)"
        parts.append(section)

    return "\n\n".join(parts)


def build_context_flat(results: list[RetrievalResult]) -> str:
    """Build a simple source-first context for free-form Q&A/API calls."""
    if not results:
        return "(Tidak ada dokumen relevan yang ditemukan.)"

    parts = []
    for idx, result in enumerate(results, 1):
        metadata = result.metadata or {}
        sumber = metadata.get("sumber", "unknown")
        judul = metadata.get("judul_halaman", "")
        halaman = metadata.get("page_number", "?")
        score = f"{result.score:.4f}" if isinstance(result.score, (int, float)) else str(result.score)

        header = f"[{idx}] Sumber: {sumber} | Hal. {halaman} | Skor: {score}"
        if judul:
            header += f" | {judul}"
        parts.append(f"{header}\n{result.text.strip()}")

    return "\n\n---\n\n".join(parts)


RUPIAH_RE = re.compile(r"\bRp\.?\s*[\d.,\s]+(?:,-|,00)?", re.IGNORECASE)
RUPIAH_AMOUNT_RE = re.compile(r"\bRp\.?\s*([0-9][0-9.,\s]*)(?:,-|,00)?", re.IGNORECASE)
NOMINAL_HINT_RE = re.compile(
    r"Rp|rupiah|nominal|besaran|sebesar|senilai|tahap|per orang|per tahap",
    re.IGNORECASE,
)
NOMINAL_CONTEXT_RE = re.compile(
    r"\b(bantuan|bansos|besaran|nominal|sebesar|senilai|per orang|per tahap|"
    r"tahap|disalurkan|diterima|penerima manfaat)\b",
    re.IGNORECASE,
)
NON_BANTUAN_AMOUNT_RE = re.compile(
    r"\b(materai|bea\s+materai)\b.{0,50}\brp\b|\brp\b.{0,50}\b(materai|bea\s+materai)\b",
    re.IGNORECASE,
)


def _canonical_amount(amount: str) -> str:
    return re.sub(r"\D", "", amount)


def _display_amount(amount: str) -> str:
    compact = _canonical_amount(amount)
    if compact.isdigit():
        return f"{int(compact):,}".replace(",", ".")
    return amount


def _extract_amounts(text: str) -> list[str]:
    amounts = []
    for match in RUPIAH_AMOUNT_RE.finditer(text):
        raw = re.sub(r"\s+", "", match.group(1)).strip(".,")
        compact = _canonical_amount(raw)
        if compact:
            amounts.append(compact)
    return amounts


def _is_bansos_nominal_text(text: str) -> bool:
    if NON_BANTUAN_AMOUNT_RE.search(text):
        return False
    return bool(RUPIAH_AMOUNT_RE.search(text) and NOMINAL_CONTEXT_RE.search(text))


def _metadata_tags(metadata: dict) -> set[str]:
    tags = metadata.get("tipe_konten", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    primary = metadata.get("tipe_konten_primer")
    if primary:
        tags.append(primary)
    return {str(tag) for tag in tags}


def _nominal_candidates(text: str) -> list[str]:
    units = []
    for raw_unit in text.splitlines():
        unit = re.sub(r"\s+", " ", raw_unit).strip()
        if _is_bansos_nominal_text(unit):
            units.append(unit)

    if units:
        return units

    for raw_unit in re.split(r"(?<=[.;])\s+", text):
        unit = re.sub(r"\s+", " ", raw_unit).strip()
        if _is_bansos_nominal_text(unit):
            units.append(unit)

    if not units and _is_bansos_nominal_text(text):
        units.append(re.sub(r"\s+", " ", text).strip())
    return units


def load_official_nominal_catalog(path: Path = NOMINAL_SOURCE_PATH) -> dict[str, dict]:
    """
    Ambil nominal resmi dari JSONL hasil clean_jsonl, bukan dari kode.
    Jika Juknis berubah, regenerate JSONL lalu katalog ini ikut berubah.
    """
    if not path.exists():
        return {}

    catalog: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            metadata = row.get("metadata") or {}
            program = metadata.get("nama_bansos")
            if not program:
                continue

            text = row.get("text", "")
            tags = _metadata_tags(metadata)
            if "nominal_bantuan" not in tags and not NOMINAL_CONTEXT_RE.search(text):
                continue

            for candidate in _nominal_candidates(text):
                amounts = _extract_amounts(candidate)
                if not amounts:
                    continue

                entry = catalog.setdefault(program, {
                    "display_name": PROGRAM_DISPLAY_NAMES.get(program, program),
                    "amounts": {},
                    "summary": "",
                    "page": metadata.get("page_number", "?"),
                    "source": metadata.get("sumber", ""),
                })

                for amount in amounts:
                    entry["amounts"][amount] = _display_amount(amount)

                candidate = candidate[:620].rstrip()
                existing_amount_count = len(set(_extract_amounts(entry.get("summary", ""))))
                candidate_amount_count = len(set(amounts))
                is_better_summary = (
                    not entry["summary"]
                    or candidate_amount_count > existing_amount_count
                    or (
                        candidate_amount_count == existing_amount_count
                        and str(metadata.get("retrieval_priority", "")) == "normal"
                        and "lampiran" not in str(metadata.get("judul_halaman", "")).lower()
                        and "lampiran" in entry["summary"].lower()
                    )
                )
                if is_better_summary:
                    entry["summary"] = candidate
                    entry["page"] = metadata.get("page_number", "?")
                    entry["source"] = metadata.get("sumber", "")

    return catalog


def build_official_nominal_facts(catalog: dict[str, dict]) -> str:
    if not catalog:
        return ""

    parts = [
        "=== NOMINAL RESMI DARI JUKNIS NORMALIZED (WAJIB DIIKUTI) ===",
        "Gunakan nominal di bagian ini untuk baris 'Nominal Bantuan'.",
        "Nominal setiap program tidak boleh ditukar dengan program lain.",
        "Jika satu program memiliki beberapa nominal resmi, tulis SEMUA nominal tersebut.",
    ]
    for _, entry in catalog.items():
        amounts = ", ".join(f"Rp {amount}" for amount in entry["amounts"].values())
        summary = entry.get("summary") or amounts
        page = entry.get("page", "?")
        parts.append(
            f"\n[PROGRAM: {entry['display_name']}]\n"
            f"- Nominal resmi: {amounts}\n"
            f"- Bukti nominal: {summary} (Hal. {page})"
        )
    return "\n".join(parts)


def _program_name_from_result(result: RetrievalResult) -> str:
    src = result.metadata.get("sumber", "unknown")
    return PROGRAM_LABELS.get(src, result.metadata.get("nama_bansos", src.replace(".pdf", "")))


def _profile_desil(profil_warga: str) -> int | None:
    match = re.search(r"\bdesil\s*(\d+)\b", profil_warga, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _profile_ages(profil_warga: str) -> list[int]:
    return [int(age) for age in re.findall(r"\b(\d+)\s*tahun\b", profil_warga.lower())]


def _desil_values_from_phrase(phrase: str) -> set[int]:
    phrase = phrase.lower()
    range_match = re.search(r"(\d+)\s*(?:-|s/d|sd|sampai(?:\s+dengan)?)\s*(\d+)", phrase)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start <= end and end <= 10:
            return set(range(start, end + 1))

    values = {int(value) for value in re.findall(r"\d+", phrase)}
    return {value for value in values if 1 <= value <= 10}


def _extract_desil_requirements(text: str) -> list[tuple[set[int], str]]:
    requirements = []
    used_spans: list[tuple[int, int]] = []
    patterns = [
        r"\bdesil\s+\d+\s*(?:-|s/d|sd|sampai(?:\s+dengan)?)\s*\d+\b",
        r"\bdesil\s+\d+(?:\s*,\s*\d+)+(?:\s*(?:dan|&)\s*\d+)?\b",
        r"\bdesil\s+\d+\s*(?:dan|&)\s*\d+\b",
        r"\bdesil\s+\d+\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text.lower()):
            span = match.span()
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in used_spans):
                continue
            phrase = match.group(0)
            values = _desil_values_from_phrase(phrase)
            if values:
                requirements.append((values, re.sub(r"\s+", " ", phrase).strip(" ,.;")))
                used_spans.append(span)
    return requirements


def _extract_age_requirements(text: str) -> list[tuple[str, tuple[int | None, int | None], str]]:
    text_norm = re.sub(r"\s+", " ", text.lower())
    requirements = []

    range_re = re.compile(
        r"(?:usia|umur|berusia)[^.;]{0,60}?"
        r"(\d+)\s*(?:tahun|bulan)?\s*"
        r"(?:s/d|sd|sampai(?:\s+dengan)?|hingga|-)\s*"
        r"(?:maksimal\s*)?(\d+)\s*tahun"
    )
    for match in range_re.finditer(text_norm):
        low, high = int(match.group(1)), int(match.group(2))
        if high >= low:
            requirements.append(("range", (low, high), match.group(0).strip(" ,.;")))

    min_re = re.compile(r"(?:usia|umur|berusia|lanjut usia)[^.;]{0,40}?(\d+)\s*tahun\s+ke\s+atas")
    for match in min_re.finditer(text_norm):
        low = int(match.group(1))
        requirements.append(("min", (low, None), match.group(0).strip(" ,.;")))

    return requirements


def _age_requirement_passes(ages: list[int], kind: str, bounds: tuple[int | None, int | None]) -> bool | None:
    if not ages:
        return None
    low, high = bounds
    if kind == "range" and low is not None and high is not None:
        return any(low <= age <= high for age in ages)
    if kind == "min" and low is not None:
        return any(age >= low for age in ages)
    return None


def build_numeric_requirement_hints(
    profil_warga: str,
    results: list[RetrievalResult],
    max_per_program: int = 3,
) -> str:
    """
    Bangun petunjuk pencocokan angka dari profil vs syarat dokumen.
    Ini data-driven dari chunk retrieval, bukan aturan per-program di kode.
    """
    desil = _profile_desil(profil_warga)
    ages = _profile_ages(profil_warga)
    if desil is None and not ages:
        return ""

    hints_by_program: dict[str, list[str]] = {}
    seen = set()

    for result in results:
        src = result.metadata.get("sumber", "")
        if src not in PROGRAM_LABELS:
            continue

        tags = _metadata_tags(result.metadata)
        text = result.text
        text_lower = text.lower()
        if (
            "kriteria_penerima" not in tags
            and "desil" not in text_lower
            and "usia" not in text_lower
            and "umur" not in text_lower
        ):
            continue

        program = _program_name_from_result(result)
        page = result.metadata.get("page_number", "?")
        program_hints = hints_by_program.setdefault(program, [])

        if desil is not None:
            desil_requirements = _extract_desil_requirements(text)
            passing_desil_requirements = [
                (allowed, phrase)
                for allowed, phrase in desil_requirements
                if desil in allowed
            ]
            selected_desil_requirements = passing_desil_requirements or desil_requirements

            for allowed, phrase in selected_desil_requirements:
                key = (program, "desil", tuple(sorted(allowed)))
                if key in seen:
                    continue
                seen.add(key)
                status = "MEMENUHI" if desil in allowed else "TIDAK MEMENUHI"
                allowed_text = ", ".join(str(value) for value in sorted(allowed))
                program_hints.append(
                    f"- {program} | Hal. {page} | syarat {phrase} "
                    f"(nilai: {allowed_text}); profil Desil {desil} -> {status}."
                )
                if len(program_hints) >= max_per_program:
                    break

        if len(program_hints) < max_per_program and ages:
            for kind, bounds, phrase in _extract_age_requirements(text):
                key = (program, "usia", kind, bounds)
                if key in seen:
                    continue
                seen.add(key)
                passes = _age_requirement_passes(ages, kind, bounds)
                if passes is None:
                    continue
                status = "MEMENUHI" if passes else "TIDAK MEMENUHI"
                ages_text = ", ".join(str(age) for age in ages)
                program_hints.append(
                    f"- {program} | Hal. {page} | syarat {phrase}; "
                    f"usia pada profil: {ages_text} tahun -> {status}."
                )
                if len(program_hints) >= max_per_program:
                    break

    hints = [
        hint
        for program_hints in hints_by_program.values()
        for hint in program_hints
    ]
    if not hints:
        return ""

    return (
        "=== CEK ANGKA PROFIL VS SYARAT DOKUMEN (WAJIB DIPERHATIKAN) ===\n"
        "Bagian ini dibuat otomatis dari angka di profil dan angka pada chunk kriteria dokumen.\n"
        "Jika hasil cek angka TIDAK MEMENUHI untuk syarat wajib, status program harus TIDAK ELIGIBLE.\n"
        "Jangan pakai MUNGKIN ELIGIBLE untuk program yang memiliki satu saja hasil TIDAK MEMENUHI.\n"
        + "\n".join(hints)
    )


def build_required_program_contract() -> str:
    lines = [
        "=== KONTRAK DINAMIS 6 PROGRAM UTAMA (WAJIB) ===",
        "Daftar ini dibuat dari konfigurasi program aktif dan menjadi satu-satunya daftar program final.",
        "Setiap program di bawah WAJIB muncul tepat satu kali sebagai heading STATUS.",
        "Jangan membuat dua heading untuk program yang sama.",
        "Jika program tidak cocok dengan profil, tetap tulis sekali sebagai STATUS: TIDAK ELIGIBLE.",
        "Jangan menambah program di luar daftar ini.",
        "Sebelum menjawab, cek ulang jumlah heading STATUS: harus sama dengan jumlah program di daftar ini.",
        "",
        "DAFTAR PROGRAM DAN NAMA DOKUMEN RESMI UNTUK SITASI:",
    ]
    for idx, (source, program) in enumerate(PROGRAM_LABELS.items(), 1):
        lines.append(f"{idx}. {program} | Sumber resmi: {source}")
    return "\n".join(lines)


MAIN_PROGRAM_CHECKLIST = build_required_program_contract()


PROGRAM_STATUS_HEADING_RE = re.compile(
    r"(?im)^#{2,4}\s*(?P<title>.+?)\s*(?:â€”|—|–|-|:)\s*STATUS\s*:\s*"
    r"(?P<status>TIDAK\s+ELIGIBLE|NON[-\s]+ELIGIBLE|MUNGKIN\s+ELIGIBLE|ELIGIBLE)\b.*$"
)

FORBIDDEN_TRAILING_SECTION_RE = re.compile(
    r"(?im)^##\s*(?:Rekomendasi\s+Tindak\s+Lanjut|Catatan\s+untuk\s+Petugas\b.*|"
    r"Rekomendasi\s+Bantuan\s+Tambahan|Rekomendasi\s+Tambahan|Bantuan\s+Lain)\s*$"
)


def _normalize_program_text(text: str) -> str:
    text = text.lower()
    text = text.replace("_", " ")
    text = text.replace("â€”", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"[^a-z0-9+\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _program_aliases(program_name: str) -> set[str]:
    aliases = {program_name}
    base_name = re.sub(r"\s*\([^)]*\)", "", program_name).strip()
    if base_name:
        aliases.add(base_name)
    for inner in re.findall(r"\(([^)]*)\)", program_name):
        inner = inner.strip()
        if inner:
            aliases.add(inner)
    return {_normalize_program_text(alias) for alias in aliases if alias.strip()}


def _canonical_output_program(title: str) -> str | None:
    title_norm = _normalize_program_text(title)
    candidates: list[tuple[int, str]] = []
    for program_name in PROGRAM_LABELS.values():
        for alias in _program_aliases(program_name):
            if alias and alias in title_norm:
                candidates.append((len(alias), program_name))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def clean_rag_answer(answer: str) -> str:
    """Remove sections that violate the active 6-program output contract."""
    trailing_match = FORBIDDEN_TRAILING_SECTION_RE.search(answer)
    if trailing_match:
        answer = answer[:trailing_match.start()].rstrip()

    matches = list(PROGRAM_STATUS_HEADING_RE.finditer(answer))
    if not matches:
        return answer.strip()

    seen_programs: set[str] = set()
    delete_ranges: list[tuple[int, int]] = []
    for idx, match in enumerate(matches):
        program = _canonical_output_program(match.group("title"))
        if not program:
            continue

        if program in seen_programs:
            section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(answer)
            delete_ranges.append((match.start(), section_end))
        else:
            seen_programs.add(program)

    cleaned = answer
    for start, end in reversed(delete_ranges):
        cleaned = cleaned[:start].rstrip() + "\n\n" + cleaned[end:].lstrip()
    return cleaned.strip()


def build_nominal_facts(
    results: list[RetrievalResult],
    max_items: int = 18,
    max_per_program: int = 4,
) -> str:
    """
    Ringkas fakta nominal per program dari chunk yang mengandung angka Rp.
    Struktur per program penting agar LLM tidak menyalin nominal program lain.
    """
    facts_by_program: dict[str, list[str]] = {}
    seen = set()
    total = 0

    for r in results:
        text = r.text.strip()
        if not RUPIAH_RE.search(text):
            continue

        src = r.metadata.get("sumber", "unknown")
        program = PROGRAM_LABELS.get(src, r.metadata.get("nama_bansos", src.replace(".pdf", "")))
        page = r.metadata.get("page_number", "?")

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line and NOMINAL_HINT_RE.search(line):
                lines.append(line)
        excerpt = " ".join(lines) if lines else text
        excerpt = re.sub(r"\s+", " ", excerpt).strip()
        if len(excerpt) > 420:
            excerpt = excerpt[:420].rstrip() + "..."

        key = (program, page, excerpt[:120])
        if key in seen:
            continue
        seen.add(key)

        program_facts = facts_by_program.setdefault(program, [])
        if len(program_facts) >= max_per_program:
            continue
        program_facts.append(f"- Hal. {page}: {excerpt}")
        total += 1

        if total >= max_items:
            break

    if not facts_by_program:
        return ""

    parts = [
        "=== FAKTA NOMINAL PER PROGRAM (WAJIB DIIKUTI) ===",
        "Aturan keras nominal:",
        "- Saat mengisi Nominal Bantuan untuk satu program, gunakan hanya nominal pada blok program yang sama.",
        "- Jangan memindahkan nominal dari program lain.",
        "- Jika blok program tidak memuat nominal eksplisit, boleh cari di konteks program yang sama.",
        "- Jika tetap tidak ada nominal eksplisit, tulis '(nominal tidak tersebut di dokumen)' tanpa angka Rp.",
    ]

    for program, facts in facts_by_program.items():
        parts.append(f"\n[PROGRAM: {program}]")
        parts.extend(facts)

    return "\n".join(parts)


# ============================================================
# CONSOLE & DISPLAY
# ============================================================
console = Console()
logger  = logging.getLogger(__name__)


def silence_loggers():
    logging.getLogger().setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)


def render_header():
    header = Text()
    header.append("  SIRA RAG — MKN3", style="bold cyan")
    header.append("  ░▒▓  Rekomendasi Bantuan Sosial ░▒▓  ", style="dim cyan")
    header.append("Tim 4 UB\n", style="bold white")

    info = Text()
    info.append("  ◈ LLM        ", style="dim")
    info.append(f"{RUNPOD_MODEL_NAME}\n", style="bold green")
    info.append("  ◈ Embedding  ", style="dim")
    info.append(f"{EMBED_MODEL_NAME}\n", style="green")
    info.append("  ◈ Reranker   ", style="dim")
    info.append(f"{RERANKER_MODEL_NAME}\n", style="green")
    info.append("  ◈ Collection ", style="dim")
    info.append(f"{QDRANT_COLLECTION}", style="green")
    info.append(f"  │  Top-K={RETRIEVAL_TOP_K}  Top-N={RERANK_TOP_N}\n", style="dim")
    info.append("  ◈ Mode       ", style="dim")
    info.append("temperature=0 (deterministik)\n", style="yellow")
    info.append("\n  Masukkan profil warga (multi-baris). Baris kosong = selesai.\n", style="dim white")
    info.append("  Ketik ", style="dim white")
    info.append("exit", style="bold red")
    info.append(" untuk keluar.", style="dim white")

    console.print(Panel(
        Text.__add__(header, info),
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    ))


def render_chunk_card(rank: int, result: RetrievalResult):
    title = Text()
    title.append(f" [{rank}] ", style="bold cyan")
    title.append(f"rerank={result.score:.4f}", style="bold yellow")
    title.append(f"  embed={result.embed_score:.4f}", style="dim yellow")

    body = Text()
    src     = result.metadata.get("sumber", "unknown")
    program = PROGRAM_LABELS.get(src, src.replace(".pdf", ""))
    judul   = result.metadata.get("judul_halaman", "")
    halaman = result.metadata.get("page_number", "?")

    body.append(f"  📄 {program}\n", style="bold blue")
    if judul:
        body.append(f"  📍 {judul} (hal. {halaman})\n", style="dim italic")
    else:
        body.append(f"  📍 Hal. {halaman}\n", style="dim italic")
    body.append(f"\n  {result.text.strip()}", style="green")

    console.print(Panel(body, title=title, title_align="left",
                        border_style="dim cyan", box=box.ROUNDED, padding=(0, 1)))


# ============================================================
# RAG GENERATOR
# ============================================================

class RAGGenerator:
    """
    RAG Generator untuk MKN3.

    Retrieval : multi-query dengan filter Qdrant per dokumen program
                → cegah chunk antar program tercampur
    Generation: API model eksternal via webservice.py
    Prompt    : RANKING_SYSTEM_PROMPT dari config.py

    Importable untuk webservice.py:
        from generation import RAGGenerator
        rag = RAGGenerator()
        result = rag.recommend(profil_warga, scoring_result="...")
    """

    def __init__(self):
        with console.status("[bold cyan]⏳ Memuat model...", spinner="dots"):
            self.retriever = PolicyRetriever()
            
            if LLM_PROVIDER == "huggingface":
                from langchain_community.llms import HuggingFaceEndpoint
                self.llm = HuggingFaceEndpoint(
                    repo_id=HF_GENERATION_MODEL,
                    huggingfacehub_api_token=HF_TOKEN,
                    temperature=0.01,
                    max_new_tokens=4000,
                    stop_sequences=[
                        "## Ringkasan Profil Warga\n##",
                        "---\n\n## Ringkasan",
                    ],
                )
                self.model_name = HF_GENERATION_MODEL
            else:
                raise RuntimeError(
                    "RAGGenerator CLI lokal tidak aktif. Gunakan webservice.py "
                    "endpoint /recommend untuk generation via API RunPod."
                )
                
        console.print(
            f"[bold green]✅ Model siap.[/] [dim]({self.model_name}, "
            "temperature=0)[/]"
        )

    def _build_qdrant_filter(self, files: list[str]):
        """Buat filter Qdrant untuk membatasi retrieval ke file tertentu."""
        from qdrant_client import models
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="sumber",
                    match=models.MatchAny(any=files),
                )
            ]
        )

    def recommend(
        self,
        profil_warga: str,
        scoring_result: str = "",
        stream: bool = True,
        show_chunks: bool = True,
    ) -> str:
        """
        Ranking program bantuan berdasarkan profil warga.

        Multi-query retrieval:
        - Query per program (dengan filter file) -> cegah chunk tercampur
        - Query nominal -> pastikan besaran bantuan tertangkap
        - Semua konteks dibatasi ke 6 program utama
        """
        p_lower = profil_warga.lower()

        # ── Deteksi keterampilan untuk query KIP Jawara ──────
        skill_kw = ["jahit", "dagang", "warung", "bengkel", "ternak",
                    "tani", "masak", "kue", "ojek", "supir"]
        skills_found = [kw for kw in skill_kw if kw in p_lower]
        skill_inject = " ".join(skills_found) if skills_found else ""

        # ── Query per program ─────────────────────────────────
        # Query nominal lintas 6 program utama.
        query_nominal = (
            "nominal bantuan Rp rupiah senilai sebesar besaran dana per tahap "
            "pakta integritas surat pernyataan lampiran pencairan "
            "PKH Plus ASPD kemiskinan ekstrem KIP KPM KIP PPKS KIP Putri"
        )
        t0 = time.time()
        all_results: list[RetrievalResult] = []
        seen: set[str] = set()

        # ── Retrieval per program (dengan filter) ─────────────
        for source, program_name in PROGRAM_LABELS.items():
            query = (
                "kriteria syarat sasaran penerima nominal mekanisme pencairan "
                f"{program_name} {skill_inject} {profil_warga[:150]}"
            )
            q_filter = self._build_qdrant_filter([source])
            results = self.retriever.retrieve(
                query,
                top_k=20,
                top_n=5,
                query_filter=q_filter,
            )
            for r in results:
                key = r.text[:100]
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)

        # ── Retrieval nominal (tanpa filter — lintas program) ─
        results_nominal = self.retriever.retrieve(
            query_nominal,
            top_k=20,
            top_n=8,
            query_filter=self._build_qdrant_filter(list(PROGRAM_LABELS.keys())),
        )
        for r in results_nominal:
            key = r.text[:100]
            if key not in seen:
                seen.add(key)
                all_results.append(r)

        retrieval_time = time.time() - t0

        if not all_results:
            console.print(Panel("[yellow]Tidak ada dokumen relevan.[/]", border_style="yellow"))
            return ""

        programs_covered = len(set(
            r.metadata.get("sumber", "") for r in all_results
            if r.metadata.get("sumber", "") in PROGRAM_LABELS
        ))

        # ── Display ──────────────────────────────────────────
        console.print()
        console.print(Panel(
            Text(
                f"  PROFIL  {profil_warga[:200]}"
                f"{'...' if len(profil_warga) > 200 else ''}",
                style="bold white",
            ),
            border_style="cyan", box=box.HEAVY, padding=(0, 1),
        ))
        console.print(
            f"\n[dim]  ⏱ Retrieval: {retrieval_time:.2f}s  │  "
            f"{len(all_results)} chunk dari {programs_covered}/6 program[/]\n"
        )

        if show_chunks:
            for i, r in enumerate(all_results, 1):
                render_chunk_card(i, r)

        # ── Build context ─────────────────────────────────────
        official_nominal_catalog = load_official_nominal_catalog()
        context_parts = [MAIN_PROGRAM_CHECKLIST]
        official_nominal_facts = build_official_nominal_facts(official_nominal_catalog)
        if official_nominal_facts:
            context_parts.append(official_nominal_facts)
        numeric_requirement_hints = build_numeric_requirement_hints(profil_warga, all_results)
        if numeric_requirement_hints:
            context_parts.append(numeric_requirement_hints)
        nominal_facts = build_nominal_facts(all_results)
        if nominal_facts:
            context_parts.append(nominal_facts)
        context_parts.append(build_context_grouped(all_results))
        context = "\n\n".join(context_parts)

        # ── Profil terstruktur ────────────────────────────────
        profil_terstruktur = parse_profil(profil_warga)
        profil_section = profil_terstruktur
        if scoring_result.strip():
            profil_section += f"\n\n=== HASIL SCORING MKN1 ===\n{scoring_result}"

        # ── Susun final prompt ────────────────────────────────
        if scoring_result.strip():
            final_prompt = POLICY_PROMPT_TEMPLATE.format(
                system_prompt=RANKING_SYSTEM_PROMPT,
                scoring_result=profil_section,
                context=context,
            )
        else:
            final_prompt = PROMPT_TEMPLATE.format(
                system_prompt=RANKING_SYSTEM_PROMPT,
                context=context,
                query=profil_section,
            )

        console.print()
        if stream:
            return self._stream(final_prompt)

        return self._full(final_prompt)

    def _stream(self, prompt: str) -> str:
        full = []
        t0   = time.time()

        with console.status("[bold green]🧠 Generating...", spinner="dots"):
            for chunk in self.llm.stream(prompt):
                token = chunk if isinstance(chunk, str) else chunk.content
                full.append(token)

        answer  = clean_rag_answer("".join(full))
        elapsed = time.time() - t0

        title = Text()
        title.append("  🏆 Ranking Rekomendasi Bantuan Sosial  ", style="bold black on green")
        console.print(Panel(Text(answer, style="green"), title=title,
                            title_align="left", border_style="green",
                            box=box.DOUBLE, padding=(1, 2)))
        console.print(f"[dim]  ⏱ Generasi: {elapsed:.1f}s  │  {len(answer)} karakter[/]\n")
        return answer

    def _full(self, prompt: str) -> str:
        with console.status("[bold green]🧠 Generating...", spinner="dots"):
            t0       = time.time()
            response = self.llm.invoke(prompt)
            elapsed  = time.time() - t0

        answer = response if isinstance(response, str) else response.content
        answer = clean_rag_answer(answer)

        title = Text()
        title.append("  🏆 Ranking Rekomendasi Bantuan Sosial  ", style="bold black on green")
        console.print(Panel(Text(answer, style="green"), title=title,
                            title_align="left", border_style="green",
                            box=box.DOUBLE, padding=(1, 2)))
        console.print(f"[dim]  ⏱ {elapsed:.1f}s  │  {len(answer)} karakter[/]")
        return answer


# ============================================================
# INTERACTIVE CLI
# ============================================================

def interactive_cli():
    silence_loggers()
    console.clear()
    render_header()
    console.print()

    rag = RAGGenerator()
    console.print()

    while True:
        try:
            console.print(Rule(style="dim cyan"))
            console.print("[dim]  Masukkan profil warga (multi-baris OK).[/]")
            console.print("[dim]  Baris kosong = selesai input. 'exit' = keluar.\n[/]")

            lines = []
            while True:
                try:
                    line = console.input("[cyan]  > [/]")
                except EOFError:
                    break

                if line.strip().lower() in ("exit", "quit", "q", "keluar"):
                    console.print(Panel(
                        "[bold white]👋 Terima kasih!\n[dim]   Tim MKN4 UB[/]",
                        border_style="cyan", box=box.DOUBLE_EDGE, padding=(1, 2),
                    ))
                    return

                if line.strip() == "" and lines:
                    break
                if line.strip():
                    lines.append(line)

            profil = "\n".join(lines).strip()
            if not profil:
                console.print("[dim yellow]  ⚠ Profil kosong, coba lagi.[/]")
                continue

            console.print("\n[dim]  Hasil scoring MKN1? (Enter jika tidak ada)[/]")
            try:
                scoring = console.input("[cyan]  Scoring › [/]").strip()
            except EOFError:
                scoring = ""

            rag.recommend(profil, scoring_result=scoring, stream=True, show_chunks=True)

        except KeyboardInterrupt:
            console.print("\n[bold cyan]👋 Dihentikan.[/]")
            break
        except Exception as e:
            console.print(f"\n[bold red]  ❌ Error: {e}[/]\n")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/]")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    configure_utf8_stdio()
    interactive_cli()
