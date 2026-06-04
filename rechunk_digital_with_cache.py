import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import hashlib

# Configuration
CHUNK_SIZE = 2000     # New chunk size
CHUNK_OVERLAP = 500   # New chunk overlap
PAGE_OVERLAP_CHARS = 300
TEXT_TO_ELEMENT_RATIO_THRESHOLD = 50

INPUT_DIR = "pdf_juknis"
OUTPUT_DIR = "chunked_data"
OLD_JSONL = os.path.join(OUTPUT_DIR, "juknis_extracted.jsonl")
NEW_JSONL = os.path.join(OUTPUT_DIR, "juknis_extracted_new.jsonl")

HARD_CAP_CHARS = 2000  # Batas karakter mutlak per chunk

def clean_backticks(text: str) -> str:
    """
    Hapus artefak backtick markdown (```) dari output Vision LLM.
    Token ini adalah noise bagi model embedding BGE-M3.
    """
    # Hapus fence block markdown: ```markdown ... ``` atau hanya ```
    text = re.sub(r'```(?:markdown|json|text|python)?\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)
    return text.strip()


def hard_cap_chunk(text: str, cap: int = HARD_CAP_CHARS) -> list[str]:
    """
    Potong teks yang melebihi `cap` karakter.
    Selalu potong di posisi \n terdekat sebelum batas — tidak pernah di tengah kalimat.
    Mengembalikan list (bisa lebih dari 1 elemen jika teks sangat panjang).
    """
    if len(text) <= cap:
        return [text]

    parts = []
    while len(text) > cap:
        # Cari newline terdekat sebelum batas cap
        cut = text.rfind('\n', 0, cap)
        if cut <= 0:
            # Tidak ada newline sama sekali — potong di spasi terdekat
            cut = text.rfind(' ', 0, cap)
        if cut <= 0:
            # Terpaksa potong tepat di cap
            cut = cap
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def make_chunk_hash(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def get_page_tail(text: str, max_chars: int = PAGE_OVERLAP_CHARS) -> str:
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

def is_complex_page(text_len: int, num_tables: int, num_drawings: int, num_images: int) -> list:
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

def load_vision_cache(jsonl_path):
    cache = {}
    if not os.path.exists(jsonl_path):
        print(f"⚠️ File {jsonl_path} tidak ditemukan. Tidak dapat memuat cache Vision.")
        return cache
    
    print(f"📦 Memuat cache Vision dari {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                meta = entry.get("metadata", {})
                if meta.get("metode") == "Vision":
                    pdf_name = meta.get("sumber")
                    page_num = meta.get("page_number")
                    judul = meta.get("judul_halaman")
                    text = entry.get("text")
                    cache[(pdf_name, page_num)] = (judul, text)
            except Exception:
                pass
    print(f"   Selesai! Berhasil memuat {len(cache)} halaman Vision ke cache.")
    return cache

def main():
    vision_cache = load_vision_cache(OLD_JSONL)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠ Tidak ada file PDF di '{INPUT_DIR}'.")
        return

    seen_hashes = set()
    stats = {"Digital": 0, "Vision": 0, "Chunks": 0, "Duplicates": 0, "CacheMiss": 0}

    print(f"🚀 Memulai re-chunking cepat (Digital + Cached Vision)...")
    print(f"   Chunk size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    with open(NEW_JSONL, "w", encoding="utf-8") as out_f:
        for pdf_name in pdf_files:
            pdf_path = os.path.join(INPUT_DIR, pdf_name)
            print(f"\n📄 Memproses: {pdf_name}")

            try:
                doc = fitz.open(pdf_path)
                prev_page_tail = ""
                
                for i, page in enumerate(tqdm(doc, desc="  Progres", unit="pg")):
                    page_num = i + 1
                    digital_text = page.get_text().strip()
                    text_len = len(digital_text)

                    tables_result = page.find_tables()
                    num_tables = len(tables_result.tables) if tables_result else 0
                    num_drawings = len(page.get_drawings())
                    num_images = len(page.get_images())

                    reasons = is_complex_page(text_len, num_tables, num_drawings, num_images)
                    is_complex = len(reasons) > 0

                    if not is_complex:
                        method = "Digital"
                        if prev_page_tail:
                            full_text = prev_page_tail + "\n" + digital_text
                        else:
                            full_text = digital_text

                        judul_hal = digital_text.split('\n')[0][:100]
                        chunks = text_splitter.split_text(full_text)

                        global_chunk_idx = 0
                        for raw_chunk in chunks:
                            # Hard cap + clean backtick untuk setiap chunk
                            sub_chunks = hard_cap_chunk(raw_chunk)
                            for chunk_text in sub_chunks:
                                chunk_text = clean_backticks(chunk_text)
                                if not chunk_text:
                                    continue
                                chunk_hash = make_chunk_hash(chunk_text)
                                if chunk_hash in seen_hashes:
                                    stats["Duplicates"] += 1
                                    continue
                                seen_hashes.add(chunk_hash)
                                global_chunk_idx += 1

                                entry = {
                                    "text": chunk_text,
                                    "metadata": {
                                        "sumber": pdf_name,
                                        "judul_halaman": judul_hal,
                                        "page_number": page_num,
                                        "chunk_index": global_chunk_idx,
                                        "total_chunks": len(chunks),
                                        "kategori": "Petunjuk Teknis (Juknis)",
                                        "metode": method
                                    }
                                }
                                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                stats["Chunks"] += 1

                        prev_page_tail = get_page_tail(digital_text)
                        stats["Digital"] += 1

                    else:
                        method = "Vision"
                        cache_key = (pdf_name, page_num)
                        
                        if cache_key in vision_cache:
                            judul_hal, konten_hal = vision_cache[cache_key]

                            # Bersihkan backtick dari output Vision LLM
                            konten_hal = clean_backticks(konten_hal)

                            # Terapkan hard cap: chunk Vision yang panjang dipecah
                            vision_sub_chunks = hard_cap_chunk(konten_hal)

                            for v_idx, v_chunk in enumerate(vision_sub_chunks):
                                if not v_chunk.strip():
                                    continue
                                chunk_hash = make_chunk_hash(v_chunk)
                                if chunk_hash not in seen_hashes:
                                    seen_hashes.add(chunk_hash)

                                    entry = {
                                        "text": v_chunk,
                                        "metadata": {
                                            "sumber": pdf_name,
                                            "judul_halaman": judul_hal,
                                            "page_number": page_num,
                                            "chunk_index": v_idx + 1,
                                            "total_chunks": len(vision_sub_chunks),
                                            "kategori": "Petunjuk Teknis (Juknis)",
                                            "metode": method
                                        }
                                    }
                                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                    stats["Chunks"] += 1
                                else:
                                    stats["Duplicates"] += 1
                            stats["Vision"] += 1
                        else:
                            print(f"  ⚠️ Cache miss untuk halaman Vision {page_num} pada {pdf_name}. Menggunakan fallback digital...")
                            stats["CacheMiss"] += 1
                            if digital_text:
                                chunks = text_splitter.split_text(digital_text)
                                for chunk_idx, chunk_text in enumerate(chunks):
                                    entry = {
                                        "text": chunk_text,
                                        "metadata": {
                                            "sumber": pdf_name,
                                            "judul_halaman": digital_text.split('\n')[0][:100],
                                            "page_number": page_num,
                                            "chunk_index": chunk_idx + 1,
                                            "total_chunks": len(chunks),
                                            "kategori": "Petunjuk Teknis (Juknis)",
                                            "metode": "Digital (Fallback)"
                                        }
                                    }
                                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                    stats["Chunks"] += 1

                        prev_page_tail = ""

                doc.close()
            except Exception as e:
                print(f"❌ Error memproses {pdf_name}: {e}")

    if os.path.exists(NEW_JSONL):
        backup_file = os.path.join(OUTPUT_DIR, "juknis_extracted_old_backup.jsonl")
        if os.path.exists(OLD_JSONL):
            if os.path.exists(backup_file):
                os.remove(backup_file)
            os.rename(OLD_JSONL, backup_file)
            print(f"💾 File lama dibackup ke: {backup_file}")
        os.rename(NEW_JSONL, OLD_JSONL)
        print(f"✅ Selesai! File baru disimpan ke: {OLD_JSONL}")
        
    print("\nSTATISTIK RE-CHUNK:")
    print(f"  Digital pages  : {stats['Digital']}")
    print(f"  Vision (Cached): {stats['Vision']}")
    print(f"  Cache Miss     : {stats['CacheMiss']}")
    print(f"  Total Chunks   : {stats['Chunks']}")
    print(f"  Duplicates     : {stats['Duplicates']}")

if __name__ == "__main__":
    main()
