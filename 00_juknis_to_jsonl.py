# ============================================================
# 00_juknis_to_jsonl.py — Hybrid Vision + Digital Extraction
# untuk Dokumen Petunjuk Teknis (Juknis/SOP)
#
# Fitur Utama:
#   1. Hybrid Detection: Otomatis memilih Digital (cepat) atau
#      Vision LLM (pintar) berdasarkan analisis elemen halaman.
#   2. Smart Chunking: Halaman digital yang panjang dipecah
#      menggunakan RecursiveCharacterTextSplitter + overlap.
#   3. Safety Net: Timeout & error pada Vision LLM tidak
#      menghentikan seluruh proses.
#   4. Cross-Page Overlap: Kalimat terpotong antar halaman
#      ditangani dengan overlap teks.
#   5. [FIX] Deduplication: Hash-based dedup mencegah chunk
#      redundan akibat cross-page overlap.
#   6. [FIX] Adaptive complexity detection via text-to-element
#      ratio untuk menangkap halaman flowchart/tabel campuran.
#   7. [FIX] prev_context scope bug diperbaiki.
#   8. [FIX] flush dilakukan per-dokumen, bukan per-halaman.
#   9. [NEW] Checkpoint/Resume: Skip halaman yang sudah berhasil.
#  10. [NEW] Retry Error: Otomatis downscale gambar untuk halaman
#      yang sebelumnya gagal (fix GGML_ASSERT error di Ollama).
#  11. [NEW] Auto-Bootstrap: Checkpoint otomatis dibangun dari JSONL
#      yang sudah ada, tanpa perlu script generate_checkpoint terpisah.
# ============================================================

import os
import json
import hashlib
import fitz  # PyMuPDF
from tqdm import tqdm
import ollama
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# KONFIGURASI
# ============================================================
INPUT_DIR = "pdf_juknis"
OUTPUT_DIR = "chunked_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "juknis_extracted.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")
MODEL_VISION = "qwen2.5vl:7b"

# Chunking config untuk Mode Digital
CHUNK_SIZE = 800        # Ukuran chunk ideal untuk embedding
CHUNK_OVERLAP = 150     # Overlap antar chunk (mengatasi kalimat terpotong)

# Cross-page overlap: berapa karakter dari akhir halaman sebelumnya
# yang ditempelkan ke awal halaman berikutnya
PAGE_OVERLAP_CHARS = 200

# Threshold rasio teks-per-elemen visual.
# Halaman dengan rasio rendah (banyak gambar/drawing relatif ke teks)
# dianggap kompleks dan dikirim ke Vision.
TEXT_TO_ELEMENT_RATIO_THRESHOLD = 50

# Downscale factors untuk retry halaman error (dari besar ke kecil)
# Default render = matrix(2,2) = ~144 DPI efektif
# Retry akan mencoba matrix yang lebih kecil agar lolos GGML_ASSERT
RETRY_SCALES = [1.5, 1.0]  # urutan downscale saat retry

# ============================================================
# PROMPT VISION
# ============================================================
SYSTEM_PROMPT = (
    "Anda adalah pakar Digitalisasi Dokumen Teknis (OCR Vision Expert).\n"
    "Tugas Anda: Mengubah gambar halaman Juknis/SOP menjadi Markdown yang SANGAT DETAIL.\n\n"
    "STRATEGI EKSTRAKSI:\n"
    "1. [TABEL BIASA]: Jika ada tabel dengan isi teks, buat ulang dalam format Markdown Table. "
    "WAJIB menyalin setiap baris dan kolom. JANGAN MERANGKUM.\n"
    "2. [TABEL FLOWCHART]: Jika tabel berisi kotak-kotak dan panah (bukan teks), "
    "JANGAN buat tabel Markdown kosong. Sebaliknya, deskripsikan ALUR secara naratif: "
    "siapa melakukan apa, kepada siapa, lewat apa. "
    "Contoh: 'Dinas Sosial Provinsi mengirim SPM ke BPKAD. BPKAD menerbitkan SP2D. "
    "Dana masuk ke Rekening Bendahara Dinas Sosial via Bank Jatim. "
    "Selanjutnya dipindahbukukan ke rekening Penerima Manfaat melalui Standing Instruction.\'\n"
    "3. [DOKUMEN SCAN/FOTO]: Jika halaman berisi foto dokumen (KTP, KK, surat), "
    "ekstrak semua teks yang terlihat. Untuk KTP: NIK, Nama, TTL, Alamat, dst. "
    "Untuk Kartu Keluarga: struktur tabel header dan baris kosong tetap dideskripsikan.\n"
    "4. [FORMULIR KOSONG]: Jika halaman berisi formulir/blanko kosong, "
    "ekstrak nama formulir, field-field yang ada, dan struktur tabelnya.\n"
    "5. [AKTOR & SYARAT]: Ekstrak setiap nama instansi/pihak terlibat dan setiap butir persyaratan secara eksplisit.\n"
    "6. [COVER]: Untuk halaman sampul, ambil Judul Utama, Tahun, dan Instansi Pembuat.\n"
    "7. [INTEGRITY]: Jika ada teks yang buram, berikan tanda [?] namun tetap tuliskan kata-kata di sekitarnya.\n\n"
    "FORMAT OUTPUT (WAJIB):\n"
    "[JUDUL]: <Judul Bab/Halaman/Prosedur>\n"
    "[KONTEN]:\n"
    "<Hasil Ekstraksi Markdown>"
)

# ============================================================
# TEXT SPLITTER (untuk chunking halaman digital yang panjang)
# ============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", ", ", " "],
    length_function=len,
)


# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def load_checkpoint() -> dict:
    """
    Load checkpoint dari file JSON.
    Format: { "pdf_name": { "completed_pages": [1,2,3,...], "failed_pages": [5,11,...] } }
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict):
    """Simpan checkpoint ke file JSON (overwrite)."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def cleanup_error_entry(output_file: str, pdf_name: str, page_num: int):
    """
    Hapus entry error lama dari JSONL ketika retry berhasil.
    Entry error ditandai dengan judul_halaman == "Ekstraksi Gagal".
    """
    if not os.path.exists(output_file):
        return 0
    kept = []
    removed = 0
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                meta = entry.get("metadata", {})
                is_error_entry = (
                    meta.get("sumber") == pdf_name
                    and meta.get("page_number") == page_num
                    and meta.get("judul_halaman") == "Ekstraksi Gagal"
                )
                if is_error_entry:
                    removed += 1
                    continue
            except Exception:
                pass
            kept.append(line)
    if removed > 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(kept))
            if kept:
                f.write("\n")
        tqdm.write(f"  🧹 Hapus {removed} entry error lama dari JSONL (hal {page_num})")
    return removed


def is_page_done(checkpoint: dict, pdf_name: str, page_num: int) -> bool:
    """
    Cek apakah halaman sudah pernah berhasil diproses.
    page_num: 1-based index.
    """
    return page_num in checkpoint.get(pdf_name, {}).get("completed_pages", [])


def is_page_failed(checkpoint: dict, pdf_name: str, page_num: int) -> bool:
    """
    Cek apakah halaman pernah gagal (masuk daftar failed_pages).
    page_num: 1-based index.
    """
    return page_num in checkpoint.get(pdf_name, {}).get("failed_pages", [])


def mark_page_done(checkpoint: dict, pdf_name: str, page_num: int):
    """Tandai halaman sebagai berhasil dan hapus dari failed_pages jika ada."""
    if pdf_name not in checkpoint:
        checkpoint[pdf_name] = {"completed_pages": [], "failed_pages": []}
    if page_num not in checkpoint[pdf_name]["completed_pages"]:
        checkpoint[pdf_name]["completed_pages"].append(page_num)
    # Hapus dari failed jika sebelumnya gagal
    if page_num in checkpoint[pdf_name].get("failed_pages", []):
        checkpoint[pdf_name]["failed_pages"].remove(page_num)


def mark_page_failed(checkpoint: dict, pdf_name: str, page_num: int):
    """Tandai halaman sebagai gagal (jika belum ada di completed)."""
    if pdf_name not in checkpoint:
        checkpoint[pdf_name] = {"completed_pages": [], "failed_pages": []}
    if page_num not in checkpoint[pdf_name].get("failed_pages", []):
        checkpoint[pdf_name]["failed_pages"].append(page_num)


# ============================================================
# CORE FUNCTIONS
# ============================================================

def get_page_image(page, scale: float = 2.0):
    """
    Render halaman ke bytes PNG.
    scale=2.0 = default (sama persis dengan kode asli matrix(2,2)).
    scale=1.5 atau 1.0 = downscale untuk retry halaman error.
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return pix.tobytes("png")


def get_page_tail(text: str, max_chars: int = PAGE_OVERLAP_CHARS) -> str:
    """
    Ambil 1-2 kalimat terakhir dari teks secara utuh (bukan blind cut).
    Menghindari pemotongan di tengah kata saat teks di-overlap ke halaman berikutnya.
    """
    if len(text) <= max_chars:
        return text

    candidate = text[-(max_chars * 2):]
    sentences = re.split(r'(?<=[.!?])\s+|\n\n|\n', candidate)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        truncated = text[-max_chars:]
        last_space = truncated.rfind(' ')
        return truncated[last_space + 1:] if last_space != -1 else truncated

    result = []
    total = 0
    for sent in reversed(sentences):
        if total + len(sent) + 1 <= max_chars:
            result.insert(0, sent)
            total += len(sent) + 1
        else:
            break

    if not result:
        result = [sentences[-1]]

    return ' '.join(result)


def make_chunk_hash(text: str) -> str:
    """Buat fingerprint SHA256 dari teks chunk (untuk deduplication)."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def is_complex_page(text_len: int, num_tables: int, num_drawings: int, num_images: int) -> list:
    """
    Deteksi halaman kompleks dan berikan alasannya.
    Return: List alasan (jika kosong, berarti halaman Digital).
    """
    reasons = []
    if num_tables > 0:
        reasons.append(f"{num_tables} Tabel")
    if num_drawings > 40:
        reasons.append(f"{num_drawings} Drawings")
    if text_len < 200:
        reasons.append(f"Teks minim ({text_len} char)")
    if text_len < 800 and num_images > 0:
        reasons.append("Gambar + Teks Sedikit")

    total_elements = num_drawings + num_images
    ratio = text_len / max(total_elements, 1)
    if total_elements > 3 and ratio < TEXT_TO_ELEMENT_RATIO_THRESHOLD:
        reasons.append(f"Rasio Rendah ({ratio:.1f})")

    return reasons


def vision_extract(img_bytes, prev_context, page_num):
    """
    Kirim gambar ke Ollama Vision LLM dengan Safety Net.
    Jika gagal/timeout, kembalikan fallback text agar proses tidak berhenti.
    """
    try:
        response = ollama.generate(
            model=MODEL_VISION,
            prompt=f"KONTEKS SEBELUMNYA: {prev_context[:300]}\n\n{SYSTEM_PROMPT}",
            images=[img_bytes],
            stream=False,
            options={"temperature": 0}
        )
        raw = response.get('response', '')

        m = re.search(r"\[JUDUL\]:\s*(.*)", raw)
        judul = m.group(1).strip() if m else "Tabel/Diagram"
        konten = raw.split("[KONTEN]:")[-1].strip()

        return judul, konten

    except Exception as e:
        print(f"  ⚠️  Vision gagal pada hal {page_num}: {e}")
        return (
            "Ekstraksi Gagal",
            f"[Halaman {page_num}] Gagal memuat menggunakan Vision LLM: {e}"
        )


def get_page_image_cropped(page, scale: float = 1.0):
    """
    Strategi crop: split halaman jadi beberapa bagian kecil.
    - Portrait (tinggi > lebar): split atas-bawah
    - Landscape/square: split kiri-kanan
    - Selalu crop tanpa cek aspek ratio — dipakai saat semua scale gagal.
    Return: list of img_bytes (2 bagian).
    """
    w = page.rect.width
    h = page.rect.height

    if h >= w:
        # Portrait atau square: split atas-bawah
        clips = [
            fitz.Rect(0, 0, w, h / 2),
            fitz.Rect(0, h / 2, w, h),
        ]
        labels = ["atas", "bawah"]
    else:
        # Landscape: split kiri-kanan
        clips = [
            fitz.Rect(0, 0, w / 2, h),
            fitz.Rect(w / 2, 0, w, h),
        ]
        labels = ["kiri", "kanan"]

    result = []
    for clip in clips:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
        result.append(pix.tobytes("png"))
    return result


def vision_extract_with_retry(page, prev_context, page_num, skip_default_scale=False):
    """
    Coba Vision extract dengan scale default dulu.
    Jika gagal (GGML_ASSERT atau error lain), retry dengan:
      1. Downscale bertahap (1.5x, 1.0x)
      2. Crop split atas-bawah (untuk halaman scan foto/dokumen portrait tinggi)

    Args:
        skip_default_scale: Jika True (halaman yang SUDAH DIKETAHUI gagal),
                            langsung mulai dari RETRY_SCALES tanpa coba 2.0x lagi.
    Return: (judul, konten, sukses: bool)
    """
    if not skip_default_scale:
        img_bytes = get_page_image(page, scale=2.0)
        judul, konten = vision_extract(img_bytes, prev_context, page_num)
        if judul != "Ekstraksi Gagal":
            return judul, konten, True
        tqdm.write(f"  ⚠️  Scale 2.0x gagal, mulai downscale retry...")
    else:
        tqdm.write(f"  ⏩ Skip scale 2.0x (diketahui gagal), langsung downscale retry...")
        judul, konten = "Ekstraksi Gagal", ""

    # Retry 1: Downscale bertahap
    for scale in RETRY_SCALES:
        tqdm.write(f"  🔄 Retry hal {page_num} dengan scale={scale}x...")
        img_bytes = get_page_image(page, scale=scale)
        judul, konten = vision_extract(img_bytes, prev_context, page_num)
        if judul != "Ekstraksi Gagal":
            tqdm.write(f"  ✅ Retry berhasil dengan scale={scale}x")
            return judul, konten, True

    # Retry 2: Crop split — selalu aktif, handles portrait DAN landscape
    tqdm.write(f"  ✂️  Retry hal {page_num} dengan crop split...")
    img_parts = get_page_image_cropped(page, scale=1.0)
    konten_parts = []
    all_ok = True
    for idx, img_part in enumerate(img_parts):
        label = f"bagian-{idx+1}"
        tqdm.write(f"     🔍 Memproses {label}...")
        j, k = vision_extract(img_part, prev_context, page_num)
        if j == "Ekstraksi Gagal":
            all_ok = False
            break
        konten_parts.append(f"[{label.upper()}]\n{k}")
    if all_ok:
        judul = "Halaman Split"
        konten = "\n\n".join(konten_parts)
        tqdm.write(f"  ✅ Crop split berhasil")
        return judul, konten, True

    # Retry 3: Fallback Digital — ekstrak teks mentah PyMuPDF tanpa Vision
    # Berguna untuk halaman formulir/blanko yang teksnya bisa dibaca langsung
    tqdm.write(f"  📄 Retry hal {page_num} dengan fallback Digital (PyMuPDF)...")
    digital_text = page.get_text().strip()
    if len(digital_text) >= 50:
        tqdm.write(f"  ✅ Fallback Digital berhasil ({len(digital_text)} char)")
        return "Fallback Digital", digital_text, True

    tqdm.write(f"  ❌ Semua retry gagal untuk hal {page_num}")
    return judul, konten, False


def main():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"📁 Folder '{INPUT_DIR}' dibuat. Masukkan file PDF Juknis di sana.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠  Tidak ada file PDF di '{INPUT_DIR}'.")
        return

    # Load checkpoint dari run sebelumnya
    checkpoint = load_checkpoint()

    # ── AUTO-BOOTSTRAP ────────────────────────────────────────────
    # Kalau checkpoint belum ada TAPI JSONL sudah ada,
    # bangun checkpoint otomatis dari JSONL (tidak perlu script terpisah).
    # Ini menangani kasus: run pertama pakai kode lama (tanpa checkpoint),
    # lalu upgrade ke kode baru ini.
    if not checkpoint and os.path.exists(OUTPUT_FILE):
        print(f"⚙️  Checkpoint belum ada tapi JSONL ditemukan → Auto-bootstrap checkpoint dari JSONL...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    meta = entry.get("metadata", {})
                    pdf_name_b = meta.get("sumber", "")
                    page_num_b = meta.get("page_number", 0)
                    if pdf_name_b and page_num_b:
                        if pdf_name_b not in checkpoint:
                            checkpoint[pdf_name_b] = {"completed_pages": [], "failed_pages": []}
                        if page_num_b not in checkpoint[pdf_name_b]["completed_pages"]:
                            checkpoint[pdf_name_b]["completed_pages"].append(page_num_b)
                except Exception:
                    pass
        save_checkpoint(checkpoint)
        total_bootstrapped = sum(len(v["completed_pages"]) for v in checkpoint.values())
        print(f"   ✅ Bootstrap selesai: {total_bootstrapped} halaman tercatat sebagai selesai.")
    # ────────────────────────────────────────────────────────────

    # Hitung berapa halaman yang perlu diproses ulang
    total_failed_retry = 0
    for pdf_name in pdf_files:
        cp = checkpoint.get(pdf_name, {})
        failed = cp.get("failed_pages", [])
        total_failed_retry += len(failed)

    if total_failed_retry > 0:
        print(f"🔁 Mode RESUME: Ditemukan {total_failed_retry} halaman gagal dari run sebelumnya → akan di-retry.")
    elif checkpoint:
        print(f"🔁 Mode RESUME: Checkpoint ditemukan → halaman yang sudah selesai akan di-skip.")
    else:
        print(f"🆕 Mode BARU: Tidak ada checkpoint dan JSONL, mulai dari awal.")

    stats = {"Digital": 0, "Vision": 0, "Chunks": 0, "Duplicates": 0, "Errors": 0, "Skipped": 0, "Retried": 0, "RetryOK": 0}

    # Load hashes dari JSONL yang sudah ada (agar dedup tetap konsisten)
    seen_hashes: set[str] = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        h = make_chunk_hash(entry.get("text", ""))
                        seen_hashes.add(h)
                    except Exception:
                        pass
        print(f"   📂 JSONL existing: {len(seen_hashes)} chunk hash dimuat untuk deduplication.")

    print(f"🚀 Memulai Ekstraksi HYBRID ULTIMATE untuk {len(pdf_files)} file...")
    print(f"   Chunk size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, page_overlap={PAGE_OVERLAP_CHARS}")
    print(f"   Text-to-element ratio threshold={TEXT_TO_ELEMENT_RATIO_THRESHOLD}")

    # Buka JSONL dalam mode APPEND agar output lama tidak tertimpa
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for pdf_name in pdf_files:
            pdf_path = os.path.join(INPUT_DIR, pdf_name)
            print(f"\n📄 Memproses: {pdf_name}")

            cp = checkpoint.get(pdf_name, {})
            completed_pages = set(cp.get("completed_pages", []))
            failed_pages = set(cp.get("failed_pages", []))

            # Hitung halaman yang perlu diproses untuk PDF ini
            try:
                doc_check = fitz.open(pdf_path)
                total_pages = len(doc_check)
                doc_check.close()
            except Exception as e:
                print(f"  ❌ Tidak bisa membuka {pdf_name}: {e}")
                stats["Errors"] += 1
                continue

            pages_to_process = []
            pages_to_skip = []
            for pg in range(1, total_pages + 1):
                if pg in completed_pages and pg not in failed_pages:
                    pages_to_skip.append(pg)
                else:
                    pages_to_process.append(pg)

            if not pages_to_process:
                print(f"  ✅ Semua {total_pages} halaman sudah selesai → skip PDF ini.")
                stats["Skipped"] += total_pages
                continue

            retry_pages = [pg for pg in pages_to_process if pg in failed_pages]
            new_pages = [pg for pg in pages_to_process if pg not in failed_pages]

            print(f"  📊 Total: {total_pages} hal | Skip: {len(pages_to_skip)} | Baru: {len(new_pages)} | Retry error: {len(retry_pages)}")
            stats["Skipped"] += len(pages_to_skip)

            try:
                doc = fitz.open(pdf_path)
                prev_context = ""
                prev_page_tail = ""

                for i, page in enumerate(tqdm(doc, desc="  Progres", unit="pg")):
                    page_num = i + 1  # 1-based

                    # Skip halaman yang sudah berhasil
                    if page_num in completed_pages and page_num not in failed_pages:
                        stats["Skipped"] += 0  # sudah dihitung di atas
                        continue

                    is_retry = page_num in failed_pages
                    if is_retry:
                        stats["Retried"] += 1

                    digital_text = page.get_text().strip()
                    text_len = len(digital_text)

                    tables_result = page.find_tables()
                    num_tables = len(tables_result.tables) if tables_result else 0
                    num_drawings = len(page.get_drawings())
                    num_images = len(page.get_images())

                    reasons = is_complex_page(text_len, num_tables, num_drawings, num_images)
                    is_complex = len(reasons) > 0

                    if not is_complex:
                        # ── MODE A: DIGITAL (SUPER CEPAT) ──
                        method = "Digital"

                        if prev_page_tail:
                            full_text = prev_page_tail + "\n" + digital_text
                        else:
                            full_text = digital_text

                        judul_hal = digital_text.split('\n')[0][:100]
                        chunks = text_splitter.split_text(full_text)

                        for chunk_idx, chunk_text in enumerate(chunks):
                            chunk_hash = make_chunk_hash(chunk_text)
                            if chunk_hash in seen_hashes:
                                stats["Duplicates"] += 1
                                continue
                            seen_hashes.add(chunk_hash)

                            entry = {
                                "text": chunk_text,
                                "metadata": {
                                    "sumber": pdf_name,
                                    "judul_halaman": judul_hal,
                                    "page_number": page_num,
                                    "chunk_index": chunk_idx + 1,
                                    "total_chunks": len(chunks),
                                    "kategori": "Petunjuk Teknis (Juknis)",
                                    "metode": method
                                }
                            }
                            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            stats["Chunks"] += 1

                        prev_page_tail = get_page_tail(digital_text)
                        prev_context = digital_text[:500]

                        mark_page_done(checkpoint, pdf_name, page_num)

                    else:
                        # ── MODE B: VISION ──
                        method = "Vision"
                        label = "RETRY" if is_retry else "Vision Mode"
                        tqdm.write(f"  🔍 {label} ({', '.join(reasons)})")

                        # Gabungkan prev_page_tail ke context Vision agar tabel/kalimat
                        # yang terpotong dari halaman Digital sebelumnya ikut terbawa
                        vision_context = prev_context
                        if prev_page_tail:
                            vision_context = prev_page_tail + "\n\n" + prev_context

                        judul_hal, konten_hal, sukses = vision_extract_with_retry(
                            page, vision_context, page_num,
                            skip_default_scale=is_retry
                        )

                        if not sukses:
                            # Semua Vision retry gagal — coba Digital fallback
                            # (berguna untuk halaman formulir kosong seperti RAB)
                            if digital_text and len(digital_text) > 20:
                                tqdm.write(f"  📄 Vision gagal total, fallback ke Digital untuk hal {page_num}...")
                                method = "Digital"
                                judul_hal = digital_text.split("\n")[0][:100]
                                chunks = text_splitter.split_text(digital_text)
                                for chunk_idx, chunk_text in enumerate(chunks):
                                    chunk_hash = make_chunk_hash(chunk_text)
                                    if chunk_hash in seen_hashes:
                                        stats["Duplicates"] += 1
                                        continue
                                    seen_hashes.add(chunk_hash)
                                    entry = {
                                        "text": chunk_text,
                                        "metadata": {
                                            "sumber": pdf_name,
                                            "judul_halaman": judul_hal,
                                            "page_number": page_num,
                                            "chunk_index": chunk_idx + 1,
                                            "total_chunks": len(chunks),
                                            "kategori": "Petunjuk Teknis (Juknis)",
                                            "metode": "Digital (Fallback)"
                                        }
                                    }
                                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                    stats["Chunks"] += 1
                                # Hapus entry error lama kalau ada
                                cleanup_error_entry(OUTPUT_FILE, pdf_name, page_num)
                                mark_page_done(checkpoint, pdf_name, page_num)
                                tqdm.write(f"  ✅ Digital fallback berhasil untuk hal {page_num}")
                            else:
                                stats["Errors"] += 1
                                mark_page_failed(checkpoint, pdf_name, page_num)
                        else:
                            if is_retry:
                                stats["RetryOK"] += 1
                                # Hapus entry error lama yang tersimpan dari run sebelumnya
                                cleanup_error_entry(OUTPUT_FILE, pdf_name, page_num)

                            chunk_hash = make_chunk_hash(konten_hal)
                            if chunk_hash not in seen_hashes:
                                seen_hashes.add(chunk_hash)

                                entry = {
                                    "text": konten_hal,
                                    "metadata": {
                                        "sumber": pdf_name,
                                        "judul_halaman": judul_hal,
                                        "page_number": page_num,
                                        "chunk_index": 1,
                                        "total_chunks": 1,
                                        "kategori": "Petunjuk Teknis (Juknis)",
                                        "metode": method
                                    }
                                }
                                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                stats["Chunks"] += 1
                            else:
                                stats["Duplicates"] += 1

                            prev_page_tail = ""
                            prev_context = konten_hal[:500]

                            mark_page_done(checkpoint, pdf_name, page_num)

                    stats[method] += 1

                out_f.flush()
                doc.close()

                # Simpan checkpoint setelah setiap PDF selesai
                save_checkpoint(checkpoint)

            except Exception as e:
                print(f"❌ Error fatal {pdf_name}: {e}")
                stats["Errors"] += 1
                save_checkpoint(checkpoint)  # Tetap simpan progress sejauh ini

    # ── Laporan Akhir ──
    print(f"\n{'='*50}")
    print(f"✅ EKSTRAKSI SELESAI!")
    print(f"{'='*50}")
    print(f"  ⏭️  Halaman di-skip  : {stats['Skipped']}")
    print(f"  📄 Digital pages   : {stats['Digital']}")
    print(f"  🔍 Vision pages    : {stats['Vision']}")
    print(f"  🔄 Retry attempts  : {stats['Retried']}")
    print(f"  ✅ Retry berhasil  : {stats['RetryOK']}")
    print(f"  📦 Total chunks    : {stats['Chunks']}")
    print(f"  🔁 Duplicates skip : {stats['Duplicates']}")
    print(f"  ❌ Errors          : {stats['Errors']}")
    print(f"  📍 Output          : {OUTPUT_FILE}")
    print(f"  📍 Checkpoint      : {CHECKPOINT_FILE}")
    print(f"{'='*50}")
    if stats["Errors"] > 0:
        print(f"⚠️  Masih ada {stats['Errors']} error. Jalankan ulang script untuk retry otomatis.")
    else:
        print("👉 Selanjutnya jalankan: python 04_embed_and_ingest_v2.py")


if __name__ == "__main__":
    main()