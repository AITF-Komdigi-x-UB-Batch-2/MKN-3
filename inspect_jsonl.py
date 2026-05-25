# inspect_jsonl.py — Viewer interaktif untuk juknis_extracted.jsonl
import json, textwrap
from collections import Counter

JSONL_FILE = "chunked_data/juknis_extracted.jsonl"

# ============================================================
# LOAD
# ============================================================

def load_all(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except:
                pass
    return records


# ============================================================
# DISPLAY
# ============================================================

def display(records, idx, filtered=None):
    """
    Tampilkan satu chunk.
    filtered: list index asli (dari all_records) kalau sedang dalam mode filter.
    """
    r = records[idx]
    m = r["metadata"]

    # Posisi dalam konteks aktif (filtered atau semua)
    if filtered is not None:
        pos = filtered.index(idx) + 1
        total = len(filtered)
        mode_label = " [MODE FILTER]"
    else:
        pos = idx + 1
        total = len(records)
        mode_label = ""

    print(f"\n{'='*65}")
    print(f"[{pos}/{total}]{mode_label}")
    print(f"📄 File   : {m['sumber']}")
    print(f"📃 Halaman: {m['page_number']}  |  Metode: {m['metode']}")
    print(f"🏷  Judul  : {m['judul_halaman']}")
    print(f"🔢 Chunk  : {m['chunk_index']}/{m['total_chunks']}  |  Index global: #{idx+1}")
    print(f"{'─'*65}")
    print(textwrap.fill(r['text'][:2000], width=80))
    if len(r['text']) > 2000:
        print(f"\n... [+{len(r['text'])-2000} karakter tersisa]")
    print(f"{'='*65}")


# ============================================================
# SUMMARY
# ============================================================

def summary(records):
    sources  = Counter(r["metadata"]["sumber"]      for r in records)
    methods  = Counter(r["metadata"]["metode"]       for r in records)

    print(f"\n{'='*65}")
    print(f"📊 SUMMARY — {len(records)} chunks total")
    print(f"  Metode : Digital={methods.get('Digital',0)}, Vision={methods.get('Vision',0)}")
    print(f"\n  Per file:")
    for src, cnt in sources.most_common():
        # Hitung rentang halaman per file
        pages = [r["metadata"]["page_number"]
                 for r in records if r["metadata"]["sumber"] == src]
        print(f"    {src}")
        print(f"      {cnt} chunks | hal {min(pages)}–{max(pages)} ({len(set(pages))} halaman unik)")
    print(f"{'='*65}")


def show_pages(records, pdf_name=None):
    """
    Tampilkan daftar halaman beserta jumlah chunk-nya.
    Kalau pdf_name diisi, filter per file.
    """
    filtered = [r for r in records
                if pdf_name is None or r["metadata"]["sumber"] == pdf_name]
    if not filtered:
        print(f"  ⚠️  Tidak ada data untuk file '{pdf_name}'.")
        return

    by_page = {}
    for r in filtered:
        key = (r["metadata"]["sumber"], r["metadata"]["page_number"])
        by_page.setdefault(key, []).append(r)

    print(f"\n{'─'*65}")
    header = f"{'File':<40} {'Hal':>5} {'Chunks':>7} {'Metode'}"
    print(header)
    print(f"{'─'*65}")
    for (src, pg), chunks in sorted(by_page.items(), key=lambda x: (x[0][0], x[0][1])):
        metode = chunks[0]["metadata"]["metode"]
        print(f"  {src[-38:]:<38} {pg:>5} {len(chunks):>7}   {metode}")
    print(f"{'─'*65}")
    print(f"  Total: {len(by_page)} halaman, {len(filtered)} chunks")


# ============================================================
# HELP
# ============================================================

def show_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    PERINTAH NAVIGASI                         ║
╠══════════════════════════════════════════════════════════════╣
║  n              → chunk berikutnya                           ║
║  p              → chunk sebelumnya                           ║
║  <angka>        → langsung ke chunk nomor sekian             ║
╠══════════════════════════════════════════════════════════════╣
║  FILTER / CARI                                               ║
║  f <kata>       → cari kata di semua teks                    ║
║  hal <n>        → lihat semua chunk di halaman n             ║
║  hal <n> <file> → halaman n dari file tertentu               ║
║  file <nama>    → filter semua chunk dari satu file          ║
║  reset          → kembali ke semua chunk (hapus filter)      ║
╠══════════════════════════════════════════════════════════════╣
║  INFO                                                        ║
║  pages          → daftar semua halaman & jumlah chunk-nya    ║
║  pages <file>   → daftar halaman dari file tertentu          ║
║  files          → daftar semua file PDF                      ║
║  s / summary    → ringkasan statistik                        ║
║  full           → tampilkan teks lengkap chunk aktif         ║
╠══════════════════════════════════════════════════════════════╣
║  h / help       → tampilkan bantuan ini                      ║
║  q              → keluar                                     ║
╚══════════════════════════════════════════════════════════════╝
""")


# ============================================================
# MAIN
# ============================================================

all_records = load_all(JSONL_FILE)
summary(all_records)

# State navigasi
active = list(range(len(all_records)))  # index yang sedang aktif (bisa difilter)
idx = 0  # index ke dalam all_records
in_filter = False

show_help()
display(all_records, active[idx])

while True:
    try:
        cmd = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nKeluar.")
        break

    cmd_lower = cmd.lower()

    if cmd_lower == 'q':
        break

    # ── Navigasi dasar ──────────────────────────────────────
    elif cmd_lower == 'n':
        idx = min(idx + 1, len(active) - 1)
        display(all_records, active[idx], active if in_filter else None)

    elif cmd_lower == 'p':
        idx = max(idx - 1, 0)
        display(all_records, active[idx], active if in_filter else None)

    elif cmd_lower.isdigit():
        target = int(cmd_lower) - 1
        target = max(0, min(target, len(active) - 1))
        idx = target
        display(all_records, active[idx], active if in_filter else None)

    # ── Filter: cari kata ───────────────────────────────────
    elif cmd_lower.startswith('f '):
        keyword = cmd[2:].strip().lower()
        hits = [i for i, r in enumerate(all_records)
                if keyword in r['text'].lower()
                or keyword in r['metadata'].get('judul_halaman','').lower()]
        if hits:
            active = hits
            in_filter = True
            idx = 0
            print(f"  🔍 '{keyword}' → {len(hits)} chunk ditemukan. Navigasi dengan n/p.")
            display(all_records, active[idx], active)
        else:
            print(f"  ⚠️  '{keyword}' tidak ditemukan.")

    # ── Filter: per halaman ─────────────────────────────────
    elif cmd_lower.startswith('hal '):
        parts = cmd.split()
        try:
            page_num = int(parts[1])
        except (IndexError, ValueError):
            print("  Format: hal <nomor>  atau  hal <nomor> <nama_file>")
            continue

        file_filter = ' '.join(parts[2:]) if len(parts) > 2 else None

        hits = [i for i, r in enumerate(all_records)
                if r['metadata']['page_number'] == page_num
                and (file_filter is None or file_filter.lower() in r['metadata']['sumber'].lower())]

        if hits:
            active = hits
            in_filter = True
            idx = 0
            label = f"hal {page_num}" + (f" ({file_filter})" if file_filter else "")
            print(f"  📃 {label} → {len(hits)} chunk. Navigasi dengan n/p.")
            display(all_records, active[idx], active)
        else:
            label = f"hal {page_num}" + (f" di '{file_filter}'" if file_filter else "")
            print(f"  ⚠️  Tidak ada chunk untuk {label}.")

    # ── Filter: per file ────────────────────────────────────
    elif cmd_lower.startswith('file '):
        keyword = cmd[5:].strip().lower()
        hits = [i for i, r in enumerate(all_records)
                if keyword in r['metadata']['sumber'].lower()]
        if hits:
            active = hits
            in_filter = True
            idx = 0
            print(f"  📄 File '{keyword}' → {len(hits)} chunk. Navigasi dengan n/p.")
            display(all_records, active[idx], active)
        else:
            print(f"  ⚠️  File '{keyword}' tidak ditemukan.")

    # ── Reset filter ────────────────────────────────────────
    elif cmd_lower == 'reset':
        active = list(range(len(all_records)))
        in_filter = False
        idx = 0
        print(f"  ✅ Filter dihapus. Kembali ke semua {len(all_records)} chunk.")
        display(all_records, active[idx])

    # ── Info: daftar halaman ─────────────────────────────────
    elif cmd_lower.startswith('pages'):
        parts = cmd.split(maxsplit=1)
        file_filter = parts[1] if len(parts) > 1 else None
        show_pages(all_records, file_filter)

    # ── Info: daftar file ────────────────────────────────────
    elif cmd_lower == 'files':
        files = sorted(set(r['metadata']['sumber'] for r in all_records))
        print(f"\n  📁 {len(files)} file PDF:")
        for i, f in enumerate(files, 1):
            cnt = sum(1 for r in all_records if r['metadata']['sumber'] == f)
            print(f"    {i}. {f}  ({cnt} chunks)")

    # ── Summary ──────────────────────────────────────────────
    elif cmd_lower in ('s', 'summary'):
        summary(all_records)

    # ── Tampilkan teks penuh ──────────────────────────────────
    elif cmd_lower == 'full':
        r = all_records[active[idx]]
        print(f"\n{'─'*65}")
        print(r['text'])
        print(f"{'─'*65}")

    # ── Help ─────────────────────────────────────────────────
    elif cmd_lower in ('h', 'help'):
        show_help()

    else:
        print("  Perintah tidak dikenal. Ketik 'h' untuk bantuan.")