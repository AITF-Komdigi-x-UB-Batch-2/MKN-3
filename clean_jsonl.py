# ============================================================
# clean_jsonl.py — Bersihkan juknis_extracted.jsonl untuk RAG
#
# Fix yang dilakukan:
#   1. [KRITIS] Hapus 8 entry error ("Gagal memuat...")
#   2. [KRITIS] Bersihkan markdown fence (``` dan **JUDUL**:**KONTEN**:)
#   3. [SEDANG]  Re-split Vision chunks >1500 char
#   4. [SEDANG]  Hapus duplikat teks
#   5. [PENTING] Contextual prefix: tambahkan sumber + judul di awal
#                setiap chunk agar embedding punya konteks yang cukup
#
# Metadata baru (v2):
#   6. [KRITIS]  nama_bansos  → label bersih program bansos per chunk
#                               untuk metadata filter di Qdrant
#   7. [PENTING] tipe_konten  → multi-label list, deteksi keyword-based
#                               untuk pengelompokan di build_context_grouped()
#
# Output: juknis_extracted_normalized.jsonl
# ============================================================

import json
import re
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE  = "chunked_data/juknis_extracted.jsonl"
OUTPUT_FILE = "chunked_data/juknis_extracted_normalized.jsonl"

CHUNK_SIZE    = 2000
CHUNK_OVERLAP = 500
RESPLIT_TRIGGER_CHARS = 2001  # Lebih dari CHUNK_SIZE agar hanya re-split jika benar-benar lewat hard cap
MIN_TEXT_CHARS        = 20
MIN_SUBCHUNK_CHARS    = 180
LOW_INFORMATION_CHARS = 220

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", ", ", " "],
    length_function=len,
)


# ============================================================
# METADATA MAPS
# ============================================================

# Mapping nama file → label program bansos bersih
# Dipakai untuk metadata filter di Qdrant (query_filter by nama_bansos)
NAMA_BANSOS_MAP = {
    "Juklak ASPD Tahun 202620260225_12303533_01.pdf":       "ASPD",
    "JUKNIS KEMISKINAN EKSTREM (13-1-2025)-1 (1) (2).pdf":  "Kemiskinan Ekstrem",
    "JUKNIS PKH PLUS 2026.pdf":                             "PKH Plus",
    "PETUNJUK TEKNIS KIP KPM JAWARA.pdf":                   "KIP KPM Jawara",
    "Petunjuk Teknis KIP PPKS Jawara 2026.pdf":             "KIP PPKS Jawara",
    "PETUNJUK TEKNIS KIP PUTRI JAWARA.pdf":                 "KIP Putri Jawara",
}

# Mapping nama file → label sumber pendek untuk context prefix
SOURCE_LABEL = {
    "Juklak ASPD Tahun 202620260225_12303533_01.pdf":       "Juklak ASPD 2026",
    "JUKNIS KEMISKINAN EKSTREM (13-1-2025)-1 (1) (2).pdf":  "Juknis Kemiskinan Ekstrem",
    "JUKNIS PKH PLUS 2026.pdf":                             "Juknis PKH Plus 2026",
    "PETUNJUK TEKNIS KIP KPM JAWARA.pdf":                   "Juknis KIP KPM Jawara",
    "Petunjuk Teknis KIP PPKS Jawara 2026.pdf":             "Juknis KIP PPKS Jawara 2026",
    "PETUNJUK TEKNIS KIP PUTRI JAWARA.pdf":                 "Juknis KIP Putri Jawara",
}


# ============================================================
# TIPE KONTEN — Keyword-based Multi-label Classifier
# ============================================================
#
# Setiap tipe punya dua kelompok keyword:
#   "kuat"  → bobot 2, keyword sangat spesifik ke tipe ini
#   "lemah" → bobot 1, keyword umum yang mendukung
#
# Tipe di-assign jika total skor >= THRESHOLD (default 2).
# Dengan cara ini, 1 kemunculan keyword kuat sudah cukup,
# tapi 1 keyword lemah saja tidak.
#
# Order penting untuk tipe_konten_primer (label pertama = dominan).

TIPE_KEYWORDS = {
    "kriteria_penerima": {
        "kuat": [
            "syarat penerima", "kriteria penerima", "sasaran penerima",
            "persyaratan penerima", "calon penerima", "berhak menerima",
            "tidak berhak menerima", "tidak eligible", "eligible",
            "desil 1", "desil 2", "dtks", "dtsen",
            "terdata dalam", "terdaftar dalam",
            "disabilitas berat", "lanjut usia 70", "lansia 70",
            "keluarga penerima manfaat kpm",
            "perempuan kepala keluarga", "miskin ekstrem",
        ],
        "lemah": [
            "syarat", "kriteria", "sasaran", "penerima manfaat",
            "berhak", "layak", "memenuhi", "ketentuan",
            "disabilitas", "lansia", "miskin", "tidak mampu",
            "kpm", "pkh", "dtks",
        ],
    },
    "nominal_bantuan": {
        "kuat": [
            "nominal bantuan", "besaran bantuan", "jumlah bantuan",
            "senilai rp", "sebesar rp", "rp.", "per tahap",
            "per triwulan", "per bulan", "per tahun",
            "nilai bantuan", "dana bantuan sebesar",
            "paket bantuan", "bantuan tunai",
        ],
        "lemah": [
            "rp", "rupiah", "nominal", "besaran", "jumlah",
            "dana", "anggaran", "biaya", "tahap", "triwulan",
        ],
    },
    "mekanisme_penyaluran": {
        "kuat": [
            "penyaluran dana", "pencairan dana", "rekening penerima",
            "bank penyalur", "pemindahbukuan", "sp2d", "spm",
            "standing instruction", "transfer dana",
            "buku tabungan", "pembukaan rekening",
            "rekening kolektif", "penyaluran bantuan",
        ],
        "lemah": [
            "penyaluran", "pencairan", "rekening", "bank",
            "transfer", "tabungan", "cair", "salur",
            "bank jatim", "bpkad", "bendahara",
        ],
    },
    "prosedur_verifikasi": {
        "kuat": [
            "verifikasi dan validasi", "assessment lapangan",
            "kunjungan lapangan", "verifikasi administrasi",
            "penetapan penerima", "seleksi penerima",
            "pemutakhiran data", "musyawarah desa",
            "musdes", "muskel", "sk gubernur", "sk bupati",
            "keputusan gubernur", "keputusan bupati",
        ],
        "lemah": [
            "verifikasi", "validasi", "assessment", "seleksi",
            "penetapan", "pemutakhiran", "musyawarah",
            "lapangan", "kunjungan", "pendataan", "sk",
        ],
    },
    "lampiran_formulir": {
        "kuat": [
            "lampiran", "formulir", "rab", "rencana anggaran belanja",
            "surat pernyataan", "pakta integritas",
            "surat kuasa", "proposal permohonan",
            "berita acara", "kartu keluarga", "ktp",
            "surat keterangan", "dokumen persyaratan",
        ],
        "lemah": [
            "lampiran", "formulir", "surat", "dokumen",
            "berkas", "proposal", "rab", "kk", "ktp",
            "identitas", "administrasi",
        ],
    },
    "pendahuluan_umum": {
        "kuat": [
            "latar belakang", "kata pengantar", "dasar hukum",
            "landasan hukum", "undang-undang nomor",
            "peraturan menteri", "peraturan gubernur",
            "tujuan program", "maksud dan tujuan",
            "ruang lingkup", "pengertian",
        ],
        "lemah": [
            "pendahuluan", "latar belakang", "tujuan",
            "dasar", "hukum", "peraturan", "kebijakan",
            "definisi", "istilah", "umum",
        ],
    },
}

# Skor minimum untuk suatu tipe di-assign ke chunk
TIPE_THRESHOLD = 2

# Urutan prioritas untuk tipe_konten_primer
TIPE_PRIORITY = [
    "kriteria_penerima",
    "nominal_bantuan",
    "mekanisme_penyaluran",
    "prosedur_verifikasi",
    "lampiran_formulir",
    "pendahuluan_umum",
]


def classify_tipe_konten_legacy(text: str) -> list[str]:
    """
    Klasifikasi multi-label tipe konten berdasarkan keyword scoring.

    Bobot:
      keyword "kuat"  → +2
      keyword "lemah" → +1
    Threshold: skor >= 2 → tipe di-assign.

    Return list tipe yang lolos threshold, diurutkan by TIPE_PRIORITY.
    Kalau tidak ada yang lolos → ["lainnya"].
    """
    text_lower = text.lower()
    hasil = []

    for tipe in TIPE_PRIORITY:
        config = TIPE_KEYWORDS[tipe]
        skor = 0

        for kw in config["kuat"]:
            if kw in text_lower:
                skor += 2

        # Hitung keyword lemah hanya jika belum pasti lolos
        # → efisiensi: stop early jika skor sudah >= threshold dari kuat saja
        if skor < TIPE_THRESHOLD:
            for kw in config["lemah"]:
                if kw in text_lower:
                    skor += 1
                if skor >= TIPE_THRESHOLD:
                    break

        if skor >= TIPE_THRESHOLD:
            hasil.append(tipe)

    return hasil if hasil else ["lainnya"]


INTRO_HINTS = [
    "kata pengantar", "puji syukur", "ucapan terima kasih",
    "apresiasi", "surabaya,", "kepala dinas",
]

NOMINAL_REQUIRED_PATTERNS = [
    r"\brp\.?\s*\d",
    r"\brupiah\b",
    r"\brab\b",
    r"rencana anggaran",
    r"harga satuan",
    r"jumlah biaya",
    r"besaran bantuan",
    r"nominal bantuan",
    r"nilai bantuan",
]

POLICY_SIGNAL_PATTERNS = [
    r"\brp\.?\s*\d",
    r"\bdesil\b",
    r"\bdtks\b",
    r"\bdtsen\b",
    r"\bsyarat\b",
    r"\bkriteria\b",
    r"\bsasaran\b",
    r"\bverifikasi\b",
    r"\bvalidasi\b",
    r"\bpencairan\b",
    r"\bpenyaluran\b",
    r"\brekening\b",
    r"\bsp2d\b",
    r"\bspm\b",
    r"\brab\b",
    r"\bktp\b",
    r"\bkk\b",
]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def has_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def keyword_in_text(text_lower: str, keyword: str) -> bool:
    kw = keyword.lower().strip()
    compact = re.sub(r"[^a-z0-9]", "", kw)

    # Keyword pendek seperti rab/rp/sk/kk sering muncul sebagai potongan kata
    # (contoh: surabaya mengandung "rab"). Pakai batas kata agar tidak false hit.
    if compact in {"rp", "rab", "kk", "ktp", "sk", "spm", "sp2d"}:
        if compact == "rp":
            return re.search(r"\brp\.?\b", text_lower) is not None
        return re.search(rf"\b{re.escape(compact)}\b", text_lower) is not None

    return kw in text_lower


def score_tipe_konten(text: str, tipe: str) -> tuple[int, list[str], list[str]]:
    text_lower = text.lower()
    config = TIPE_KEYWORDS[tipe]
    kuat_hits = [kw for kw in config["kuat"] if keyword_in_text(text_lower, kw)]
    lemah_hits = [kw for kw in config["lemah"] if keyword_in_text(text_lower, kw)]
    return (len(kuat_hits) * 2) + len(lemah_hits), kuat_hits, lemah_hits


def passes_tipe_guard(text: str, tipe: str, kuat_hits: list[str], lemah_hits: list[str]) -> bool:
    text_lower = text.lower()

    # Kata pengantar sering menyebut "penerima manfaat" secara umum.
    # Jangan jadikan itu bukti kriteria tanpa sinyal syarat yang spesifik.
    if tipe == "kriteria_penerima" and any(h in text_lower for h in INTRO_HINTS):
        concrete = (
            kuat_hits
            or "syarat" in text_lower
            or "kriteria" in text_lower
            or "sasaran" in text_lower
            or "desil" in text_lower
            or "dtks" in text_lower
            or "dtsen" in text_lower
        )
        if not concrete:
            return False

    # Label nominal harus punya sinyal uang/RAB yang eksplisit.
    if tipe == "nominal_bantuan" and not has_any_pattern(text, NOMINAL_REQUIRED_PATTERNS):
        return False

    # "Surat" sendirian terlalu umum; lampiran/formulir perlu sinyal administrasi kuat.
    if tipe == "lampiran_formulir" and not kuat_hits and len(lemah_hits) < 3:
        return False

    return True


def classify_tipe_konten(text: str) -> list[str]:
    """
    Klasifikasi multi-label yang lebih ketat:
      - 1 keyword kuat cukup jika lolos guard tipe.
      - Tanpa keyword kuat, butuh minimal 2 keyword lemah.
      - Guard tambahan mencegah kata pengantar/formulir umum dilabeli terlalu agresif.
    """
    hasil = []

    for tipe in TIPE_PRIORITY:
        skor, kuat_hits, lemah_hits = score_tipe_konten(text, tipe)
        if skor < TIPE_THRESHOLD:
            continue
        if not kuat_hits and len(lemah_hits) < 2:
            continue
        if not passes_tipe_guard(text, tipe, kuat_hits, lemah_hits):
            continue
        hasil.append(tipe)

    return hasil if hasil else ["lainnya"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_hash(text: str) -> str:
    norm = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()


def strip_markdown_fence(text: str) -> str:
    """
    Bersihkan artefak markdown fence dan label prompt yang tidak perlu:
      - ```markdown ... ```  → isi saja
      - ``` ... ```          → isi saja
      - **JUDUL**: xxx\n**KONTEN**:\n  → xxx\n
    """
    text = re.sub(r'^\s*```(?:markdown|json|text)?\s*\n?', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\n?\s*```\s*$', '', text.strip())
    text = re.sub(r'(?m)^\s*```(?:markdown|json|text)?\s*$', '', text)
    text = re.sub(r'(?m)^\s*```\s*$', '', text)
    text = re.sub(r'(?m)^\s*\*\*\s*$', '', text)
    text = re.sub(r'\*\*JUDUL\*\*:\s*(.*?)\n\n\*\*KONTEN\*\*:\s*\n', r'\1\n\n', text)
    text = re.sub(r'\*\*JUDUL\*\*:\s*(.*?)\n', r'\1\n', text)
    text = re.sub(r'\*\*KONTEN\*\*:\s*\n?', '', text)
    text = re.sub(r'\[JUDUL\]:\s*(.*?)\n', r'\1\n', text)
    text = re.sub(r'\[KONTEN\]:\s*\n?', '', text)
    return text.strip()


def clean_title(judul: str) -> str:
    judul = normalize_space(str(judul or ""))
    judul = re.sub(r"^[#*\s]+|[#*\s]+$", "", judul).strip()
    if re.match(r"^\d+\s*$", judul):
        return ""
    if judul.lower() in {"tabel/diagram", "ekstraksi gagal", "i", "ii", "iii", "iv", "v"}:
        return ""
    return judul


def split_text_for_rag(text: str) -> list[str]:
    raw_chunks = (
        text_splitter.split_text(text)
        if len(text) > RESPLIT_TRIGGER_CHARS
        else [text]
    )

    merged: list[str] = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and len(chunk) < MIN_SUBCHUNK_CHARS:
            merged[-1] = f"{merged[-1]}\n{chunk}"
        else:
            merged.append(chunk)

    return merged


def should_skip_subchunk(text: str) -> bool:
    clean = normalize_space(text)
    if len(clean) < MIN_TEXT_CHARS:
        return True
    if re.fullmatch(r"[\d\s.,:/()\-]+", clean):
        return True
    if len(clean) < 80 and not has_any_pattern(clean, POLICY_SIGNAL_PATTERNS):
        return True
    return False


def quality_flags(text: str, tipe_konten: list[str]) -> list[str]:
    clean = normalize_space(text)
    flags = []
    if len(clean) < LOW_INFORMATION_CHARS:
        flags.append("pendek")
    if len(clean.split()) < 20:
        flags.append("fragmen")
    if tipe_konten == ["lainnya"]:
        flags.append("tipe_tidak_terdeteksi")
    if not has_any_pattern(clean, POLICY_SIGNAL_PATTERNS):
        flags.append("sinyal_kebijakan_lemah")
    return flags


def is_error_entry(entry: dict) -> bool:
    text  = entry.get("text", "")
    judul = entry.get("metadata", {}).get("judul_halaman", "")
    return (
        "Gagal memuat menggunakan Vision LLM" in text
        or judul == "Ekstraksi Gagal"
    )


def inject_context_prefix(text: str, meta: dict) -> str:
    """
    Tambahkan prefix konteks di awal setiap chunk:
      [Sumber: Juknis PKH Plus 2026 | Hal. 7 | Sasaran Penerima]
      <isi chunk>
    """
    sumber_raw = meta.get("sumber", "")
    sumber     = SOURCE_LABEL.get(sumber_raw, sumber_raw.replace(".pdf", ""))
    halaman    = meta.get("page_number", "?")
    judul      = meta.get("judul_halaman", "").strip()

    if re.match(r"^\d+\s*$", judul):
        judul = ""
    if judul in ("Tabel/Diagram", "Ekstraksi Gagal", "**", ""):
        judul_part = ""
    else:
        judul_part = f" | {judul}"

    prefix = f"[Sumber: {sumber} | Hal. {halaman}{judul_part}]\n"

    if text.startswith("[Sumber:"):
        return text

    return prefix + text


# ============================================================
# MAIN
# ============================================================

def main():
    entries = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    print(f"📂 Input : {len(entries)} entries")

    stats = {
        "error_removed":      0,
        "fence_cleaned":      0,
        "prefix_added":       0,
        "resplit":            0,
        "resplit_new_chunks": 0,
        "tiny_skipped":       0,
        "low_priority":       0,
        "dupes_removed":      0,
        "final":              0,
        # metadata baru
        "nama_bansos_tagged": 0,
        "tipe_lainnya":       0,   # chunk yang tidak masuk tipe manapun
    }

    # Distribusi tipe konten untuk debug
    tipe_dist: dict[str, int] = {}
    tipe_primer_dist: dict[str, int] = {}

    seen_hashes  = set()
    output_entries = []

    for entry in entries:
        meta = entry.get("metadata", {}).copy()
        text = entry.get("text", "")

        # ── Fix 1: Hapus entry error ──────────────────────────
        if is_error_entry(entry):
            stats["error_removed"] += 1
            continue

        # ── Fix 2: Bersihkan markdown fence ──────────────────
        cleaned = strip_markdown_fence(text)
        if cleaned != text:
            stats["fence_cleaned"] += 1
            text = cleaned

        # ── Fix 2b: Bersihkan judul_halaman noise ────────────
        meta["judul_halaman"] = clean_title(meta.get("judul_halaman", ""))

        # Skip jika teks terlalu pendek setelah clean
        if len(text.strip()) < 20:
            continue

        # ── Metadata baru 1: nama_bansos ─────────────────────
        # Ditambahkan di level entry (sebelum resplit) karena
        # semua sub-chunk dari 1 halaman pasti milik program yang sama.
        sumber_raw = meta.get("sumber", "")
        nama_bansos = NAMA_BANSOS_MAP.get(sumber_raw, "")
        if nama_bansos:
            meta["nama_bansos"] = nama_bansos
        # Jika tidak ada mapping (misal regulasi tambahan), field tidak ditambahkan
        # sehingga query_filter by nama_bansos otomatis exclude chunk ini.

        # ── Fix 3: Re-split SEBELUM prefix ───────────────────
        raw_sub_chunks = split_text_for_rag(text)

        if len(raw_sub_chunks) > 1:
            stats["resplit"] += 1
            stats["resplit_new_chunks"] += len(raw_sub_chunks)

        # ── Per sub-chunk: prefix + tipe_konten + dedup ──────
        for idx, raw_sub in enumerate(raw_sub_chunks):
            raw_sub = raw_sub.strip()
            if not raw_sub:
                continue
            if should_skip_subchunk(raw_sub):
                stats["tiny_skipped"] += 1
                continue

            # Fix 5: Inject prefix
            prefixed = inject_context_prefix(raw_sub, meta)
            if prefixed != raw_sub:
                stats["prefix_added"] += 1

            # Fix 4: Deduplication
            h = make_hash(prefixed)
            if h in seen_hashes:
                stats["dupes_removed"] += 1
                continue
            seen_hashes.add(h)

            # Metadata baru 2: tipe_konten (multi-label, keyword-based)
            # Klasifikasi dilakukan pada teks TANPA prefix agar prefix
            # tidak mempengaruhi skor keyword (prefix isinya nama sumber, bukan konten).
            tipe_konten = classify_tipe_konten(raw_sub)

            # Statistik distribusi
            for t in tipe_konten:
                tipe_dist[t] = tipe_dist.get(t, 0) + 1
            if tipe_konten == ["lainnya"]:
                stats["tipe_lainnya"] += 1

            tipe_primer = tipe_konten[0]
            tipe_primer_dist[tipe_primer] = tipe_primer_dist.get(tipe_primer, 0) + 1
            flags = quality_flags(raw_sub, tipe_konten)
            retrieval_priority = "low" if flags else "normal"
            if retrieval_priority == "low":
                stats["low_priority"] += 1

            # Susun metadata final untuk chunk ini
            new_meta = meta.copy()
            new_meta["tipe_konten"] = tipe_konten
            new_meta["tipe_konten_primer"] = tipe_primer
            new_meta["retrieval_priority"] = retrieval_priority
            if flags:
                new_meta["quality_flags"] = flags

            # Hapus field lama yang tidak informatif (semua nilainya sama)
            new_meta.pop("kategori", None)   # selalu "Petunjuk Teknis (Juknis)"
            new_meta.pop("metode", None)      # selalu "Vision"

            if "nama_bansos" in new_meta:
                stats["nama_bansos_tagged"] += 1

            if len(raw_sub_chunks) > 1:
                new_meta["chunk_index"]  = idx + 1
                new_meta["total_chunks"] = len(raw_sub_chunks)
                new_meta["resplit"]      = True

            output_entries.append({"text": prefixed, "metadata": new_meta})

    # ── Tulis output ──────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for e in output_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    stats["final"] = len(output_entries)

    print("\nQUALITY SUMMARY")
    print(f"  Tiny/artefact chunk dilewati : {stats['tiny_skipped']}")
    print(f"  Low-priority chunk ditandai  : {stats['low_priority']}")

    # ── Laporan ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ CLEANING SELESAI → {OUTPUT_FILE}")
    print(f"{'='*60}")
    print(f"  ❌ Entry error dihapus        : {stats['error_removed']}")
    print(f"  🧹 Markdown fence dibersihkan : {stats['fence_cleaned']}")
    print(f"  🏷️  Contextual prefix ditambah : {stats['prefix_added']}")
    print(f"  ✂️  Chunk di-resplit           : {stats['resplit']} → {stats['resplit_new_chunks']} sub-chunk")
    print(f"  🔁 Duplikat dihapus           : {stats['dupes_removed']}")
    print(f"{'='*60}")
    print(f"  📦 Input  : {len(entries)} entries")
    print(f"  📦 Output : {stats['final']} entries")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"📊 METADATA BARU")
    print(f"{'='*60}")
    print(f"  🏷️  nama_bansos ter-tag        : {stats['nama_bansos_tagged']} entries")
    print(f"\n  📂 Distribusi tipe_konten (multi-label, total lebih dari {stats['final']}):")
    for tipe in [*[t for t in TIPE_PRIORITY if t in tipe_dist], "lainnya"]:
        count = tipe_dist.get(tipe, 0)
        bar   = "█" * (count // 5)
        print(f"    {tipe:25s}: {count:4d}  {bar}")

    print(f"\n  Distribusi tipe_konten_primer:")
    for tipe in [*[t for t in TIPE_PRIORITY if t in tipe_primer_dist], "lainnya"]:
        count = tipe_primer_dist.get(tipe, 0)
        bar   = "#" * (count // 5)
        print(f"    {tipe:25s}: {count:4d}  {bar}")

    print(f"\n  ⚠️  Chunk 'lainnya' (tidak ada tipe): {stats['tipe_lainnya']}")
    if stats["tipe_lainnya"] > 0:
        print(f"      → Review manual atau tambah keyword di TIPE_KEYWORDS")

    print(f"\n👉 Gunakan '{OUTPUT_FILE}' untuk proses embedding selanjutnya.")
    print(f"{'='*60}\n")

    # ── Contoh output untuk verifikasi ───────────────────────
    print("🔍 SAMPLE 3 ENTRY PERTAMA (verifikasi metadata):")
    print("-" * 60)
    for e in output_entries[:3]:
        m = e["metadata"]
        print(f"  sumber      : {m.get('sumber','')[:55]}")
        print(f"  nama_bansos : {m.get('nama_bansos', '(tidak ada)')}")
        print(f"  tipe_konten : {m.get('tipe_konten')}")
        print(f"  tipe_primer : {m.get('tipe_konten_primer')}")
        print(f"  priority    : {m.get('retrieval_priority')}")
        print(f"  judul       : {m.get('judul_halaman','(kosong)')[:60]}")
        print(f"  hal.        : {m.get('page_number')}")
        print(f"  teks[:80]   : {e['text'][:80].strip()}")
        print()


if __name__ == "__main__":
    main()
