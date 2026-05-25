# ============================================================
# reset_pages.py — Hapus halaman tertentu dari JSONL & checkpoint
#
# Gunakan kalau hasil ekstraksi halaman tertentu kurang memuaskan
# dan ingin di-run ulang dengan kode yang sudah diperbaiki.
#
# Cara pakai:
#   1. Edit bagian PAGES_TO_RESET di bawah
#   2. Jalankan: python reset_pages.py
#   3. Jalankan: python 00_juknis_to_jsonl.py
# ============================================================

import json
import os

OUTPUT_DIR = "chunked_data"
JSONL_FILE = os.path.join(OUTPUT_DIR, "juknis_extracted.jsonl")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.json")

# ============================================================
# EDIT DI SINI: halaman yang mau di-reset
# Format: {"nama_file_pdf": [nomor_halaman, ...]}
# Contoh: reset hal 11 dan 30 dari JUKNIS KEMISKINAN EKSTREM
# ============================================================
PAGES_TO_RESET = {
    "JUKNIS KEMISKINAN EKSTREM (13-1-2025)-1 (1) (2).pdf": [30],
    "Petunjuk Teknis KIP PPKS Jawara 2026.pdf": [15],
}


def main():
    if not PAGES_TO_RESET:
        print("⚠️  PAGES_TO_RESET kosong. Edit script ini dulu.")
        return

    print("📋 Halaman yang akan di-reset:")
    for pdf, pages in PAGES_TO_RESET.items():
        print(f"   {pdf}: hal {pages}")
    print()

    # ── Step 1: Filter JSONL ──
    if not os.path.exists(JSONL_FILE):
        print(f"❌ File tidak ditemukan: {JSONL_FILE}")
        return

    kept = []
    removed = 0
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                meta = entry.get("metadata", {})
                pdf_name = meta.get("sumber", "")
                page_num = meta.get("page_number", 0)

                if pdf_name in PAGES_TO_RESET and page_num in PAGES_TO_RESET[pdf_name]:
                    removed += 1
                    continue  # hapus entry ini
                kept.append(line)
            except Exception:
                kept.append(line)  # baris rusak tetap disimpan

    with open(JSONL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(kept))
        if kept:
            f.write("\n")

    print(f"✅ JSONL: {removed} chunk dihapus, {len(kept)} chunk tersisa.")

    # ── Step 2: Update checkpoint ──
    if not os.path.exists(CHECKPOINT_FILE):
        print("⚠️  Checkpoint tidak ditemukan, skip update checkpoint.")
        return

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    reset_count = 0
    for pdf_name, pages in PAGES_TO_RESET.items():
        if pdf_name not in checkpoint:
            continue
        cp = checkpoint[pdf_name]
        for pg in pages:
            # Hapus dari completed
            if pg in cp.get("completed_pages", []):
                cp["completed_pages"].remove(pg)
                reset_count += 1
            # Hapus dari failed juga (biar dianggap halaman baru)
            if pg in cp.get("failed_pages", []):
                cp["failed_pages"].remove(pg)

    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    print(f"✅ Checkpoint: {reset_count} halaman dihapus dari completed_pages.")
    print()
    print("👉 Sekarang jalankan: python 00_juknis_to_jsonl.py")
    print("   Halaman yang di-reset akan diproses ulang otomatis.")


if __name__ == "__main__":
    main()