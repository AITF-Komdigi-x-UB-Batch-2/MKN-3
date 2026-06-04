# ============================================================
# batch_retrieval.py — Batch Retrieval untuk Evaluasi RAG
# MKN3 / Tim 4 Universitas Brawijaya
#
# Input  : sample_retrieval_bansos_final.jsonl
#          (format OpenAI fine-tuning: messages[user] = profil warga)
# Output : retrieval_results.jsonl
#          (satu baris per warga, berisi chunks hasil retrieval beserta metadata)
#
# Cara pakai:
#   python batch_retrieval.py
#   python batch_retrieval.py --input custom_input.jsonl --output custom_out.jsonl
#   python batch_retrieval.py --no-filter   (tanpa filter PROGRAM_LABELS)
# ============================================================

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from config import (
    RERANK_TOP_N,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K,
    configure_utf8_stdio,
)

configure_utf8_stdio()

from retrieval import PolicyRetriever, RetrievalResult
from generation import PROGRAM_LABELS  # key = filename PDF, value = nama program

# ============================================================
# KONFIGURASI DEFAULT
# ============================================================

DEFAULT_INPUT = "sample_retrieval_bansos_final.jsonl"
DEFAULT_OUTPUT = "retrieval_results_minilm2.jsonl"
DEFAULT_RERANKER_MODEL = RERANKER_MODEL_NAME

# Gunakan profil warga mentah sebagai query utama retrieval.
# Jika True, juga tambahkan query spesifik per program (multi-query).
MULTI_QUERY = True

# Top-K dan Top-N bisa di-override via argumen CLI.
# Multi-query batch mengambil kandidat semantic dari beberapa query, lalu
# hanya melakukan satu final rerank. 12 x 3 query sudah lebih luas dari
# default lama 7 x 1 query tanpa membuat evaluasi terlalu lambat.
DEFAULT_TOP_K = max(RETRIEVAL_TOP_K, 12)
DEFAULT_TOP_N = RERANK_TOP_N

# Drop tail chunks yang sudah dinilai rendah oleh final reranker.
DEFAULT_SCORE_THRESHOLD = 0.2
MIN_FINAL_CHUNKS = 2

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# HELPER: Ekstrak konten dari format JSONL fine-tuning
# ============================================================


def extract_profil_from_messages(messages: list[dict]) -> dict:
    """
    Ekstrak profil warga, NIK, label ground-truth, dan skor dari pesan.

    Format input tiap entry:
      messages[0] = system prompt (diabaikan untuk query)
      messages[1] = user  → profil warga (teks mentah)
      messages[2] = assistant → laporan_evaluasi JSON (ground truth)
    """
    user_content = ""
    assistant_json = None

    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            user_content = msg.get("content", "")
        elif role == "assistant":
            raw = msg.get("content", "")
            try:
                assistant_json = json.loads(raw)
            except Exception:
                assistant_json = None

    # Ambil NIK dari baris profil
    nik = ""
    for line in user_content.splitlines():
        if "NIK" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                nik = parts[1].strip()
            break

    # Ambil label ground truth dari assistant JSON
    gt_pkh = None
    gt_aspd = None
    skor_pkh = None
    skor_aspd = None

    if assistant_json:
        evaluasi = assistant_json.get("laporan_evaluasi", {})
        kesimpulan = evaluasi.get("kesimpulan", {})
        gt_pkh = kesimpulan.get("pkh_plus", {}).get("label")
        gt_aspd = kesimpulan.get("aspd", {}).get("label")
        skor = evaluasi.get("skor", {})
        skor_pkh = skor.get("skor_pkh_plus")
        skor_aspd = skor.get("skor_aspd")

    # Ekstrak hanya bagian "Profil Warga" (baris "-" key: value) tanpa
    # paragraf instruksi ("Skor Prioritas...", "Tolong buatkan...") di
    # bagian bawah, agar profil_text yang dipakai sebagai query reranker
    # tidak terlalu panjang dan tidak membingungkan model.
    profil_lines = []
    for line in user_content.splitlines():
        stripped = line.strip()
        # Berhenti saat menemukan baris kosong setelah blok profil,
        # atau baris yang bukan data profil (dimulai bukan dengan "-")
        if stripped.startswith("Skor Prioritas") or stripped.startswith("Tolong"):
            break
        profil_lines.append(line)
    profil_text_clean = "\n".join(profil_lines).strip()

    return {
        "nik": nik,
        "profil_text": profil_text_clean,  # hanya data profil, tanpa instruksi
        "profil_text_full": user_content,  # simpan teks lengkap jika dibutuhkan
        "gt_label_pkh_plus": gt_pkh,
        "gt_label_aspd": gt_aspd,
        "skor_pkh_plus": skor_pkh,
        "skor_aspd": skor_aspd,
    }


# ============================================================
# HELPER: Bangun query strings untuk satu profil warga
# ============================================================


def build_queries(profil_text: str, multi_query: bool = MULTI_QUERY) -> list[str]:
    """
    Bangun daftar query retrieval dari teks profil warga.
    Query pertama selalu query profil terstruktur. Jika multi_query aktif,
    tambahkan query program-spesifik untuk memperluas candidate pool; semua
    kandidat akan direrank ulang terhadap query profil sebelum output final.
    """
    from webservice import _parse_content_to_retrieval_query

    parsed_query = _parse_content_to_retrieval_query(profil_text)
    queries = [parsed_query]

    if multi_query:
        queries.extend(
            [
                (
                    "syarat kriteria sasaran penerima bantuan sosial ASPD "
                    "asistensi sosial penyandang disabilitas Jawa Timur usia "
                    "6 bulan sampai maksimal 60 tahun desil 1-5 DTSEN "
                    "kesulitan aktivitas sehari-hari"
                ),
                (
                    "syarat kriteria sasaran penerima bantuan sosial PKH Plus "
                    "Jawa Timur lanjut usia 70 tahun ke atas desil 1 2 3 4 "
                    "DTSEN KTP NIK KK wilayah Jawa Timur"
                ),
            ]
        )

    # Jaga urutan, tetapi hilangkan duplikat jika parser menghasilkan teks sama.
    return list(dict.fromkeys(q for q in queries if q.strip()))


# ============================================================
# CORE: Retrieval per warga
# ============================================================


def retrieve_for_profil(
    retriever: PolicyRetriever,
    profil_info: dict,
    allowed_sources: list[str] | None,
    top_k: int | None,
    top_n: int | None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    multi_query: bool = MULTI_QUERY,
) -> dict:
    """
    Jalankan retrieval untuk satu profil warga.
    Kembalikan dict lengkap yang siap ditulis ke output JSONL.

    [Fix] Multi-query pooling bug:
    Sebelumnya, chunk dari query statis ("kriteria penerima ASPD...",
    "kriteria penerima PKH Plus...") mendapat rerank score tinggi secara
    konstan (mis. 0.968) karena cross-encoder menilai relevansi
    (query_statis, chunk_program) — BUKAN (profil_warga, chunk).
    Setelah pooling + dedup, chunk dengan score tertinggi dari query
    statis selalu mendominasi ranking akhir, terlepas dari profil warga.

    Fix: Setelah semua chunk unik terkumpul dari multi-query, lakukan
    final reranking ulang terhadap `profil_text` sehingga ranking akhir
    mencerminkan relevansi nyata ke profil warga.
    """
    nik = profil_info["nik"]
    profil_text = profil_info["profil_text"]
    queries = build_queries(profil_text, multi_query=multi_query)
    final_query = queries[0]
    final_top_n = top_n or retriever.default_top_n

    logger.info("=" * 60)
    logger.info("🔎 NIK: %s | %d query", nik, len(queries))

    # Kumpulkan kandidat semantic dari semua query, de-dup berdasarkan text.
    # Reranking hanya dilakukan sekali di bawah terhadap query profil.
    results_by_text: dict[str, RetrievalResult] = {}

    for q_idx, query in enumerate(queries, 1):
        q_preview = query[:80].replace("\n", " ")
        logger.info("   Query %d/%d: %s...", q_idx, len(queries), q_preview)

        try:
            results = retriever.semantic_search(
                query,
                top_k=top_k,
                allowed_sources=allowed_sources,
            )
        except Exception as e:
            logger.error("   ❌ Query %d gagal: %s", q_idx, e)
            results = []

        for r in results:
            r.embed_score = r.score
            existing = results_by_text.get(r.text)
            if existing is None or r.embed_score > existing.embed_score:
                results_by_text[r.text] = r

    all_results = list(results_by_text.values())

    # Final rerank wajib dilakukan terhadap query profil. Tanpa tahap ini,
    # skor dari query statis ASPD/PKH dapat mendominasi ranking akhir.
    if all_results:
        semantic_scores = {r.text: r.embed_score for r in all_results}
        all_results = retriever.rerank(
            final_query,
            all_results,
            top_n=len(all_results),
        )
        for r in all_results:
            r.embed_score = semantic_scores.get(r.text, r.embed_score)

    # Filter tail context setelah final rerank. Jangan perlakukan semua chunk
    # yang punya tag lampiran_formulir sebagai lampiran murni, karena beberapa
    # chunk kriteria juga membawa tag itu dari proses ekstraksi.
    n_before = len(all_results)
    final_results: list[RetrievalResult] = []
    important_types = {"kriteria_penerima", "nominal_bantuan"}

    for r in all_results:
        tipe_konten = set(r.metadata.get("tipe_konten", []))
        primary_type = r.metadata.get("tipe_konten_primer")
        is_lampiran_only = (
            "lampiran_formulir" in tipe_konten
            and primary_type == "lampiran_formulir"
            and not tipe_konten.intersection(important_types)
        )
        is_low_priority = r.metadata.get("retrieval_priority") == "low"

        if is_lampiran_only:
            continue
        if is_low_priority and r.score < score_threshold:
            continue
        if r.score < score_threshold and len(final_results) >= MIN_FINAL_CHUNKS:
            continue

        final_results.append(r)
        if len(final_results) >= final_top_n:
            break

    all_results = final_results

    n_filtered = n_before - len(all_results)
    if n_filtered:
        logger.info(
            "   🗑️ Filter final noise: %d dibuang, %d tersisa (threshold=%.2f).",
            n_filtered,
            len(all_results),
            score_threshold,
        )

    logger.info(
        "   ✅ Total chunk unik: %d (dari %d query)",
        len(all_results),
        len(queries),
    )

    return {
        "nik": nik,
        "gt_label_pkh_plus": profil_info["gt_label_pkh_plus"],
        "gt_label_aspd": profil_info["gt_label_aspd"],
        "skor_pkh_plus": profil_info["skor_pkh_plus"],
        "skor_aspd": profil_info["skor_aspd"],
        "num_queries": len(queries),
        "num_chunks": len(all_results),
        "retrieved_chunks": [r.to_dict() for r in all_results],
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch retrieval dari sample JSONL fine-tuning."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"File JSONL input (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"File JSONL output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Override top-K untuk semantic search",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Override top-N untuk reranking",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Model cross-encoder reranker (default: {DEFAULT_RERANKER_MODEL})",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Buang chunk final dengan rerank score di bawah nilai ini (default: {DEFAULT_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--single-query",
        action="store_true",
        help="Nonaktifkan query program-spesifik; hanya pakai query profil terstruktur",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Nonaktifkan filter PROGRAM_LABELS (retrieval dari seluruh collection)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("❌ File input tidak ditemukan: %s", input_path)
        sys.exit(1)

    # Tentukan allowed_sources
    allowed_sources: list[str] | None = None
    if not args.no_filter:
        allowed_sources = list(PROGRAM_LABELS.keys())
        logger.info(
            "🔒 Filter aktif: %d program dari PROGRAM_LABELS", len(allowed_sources)
        )
    else:
        logger.info("🔓 Filter dinonaktifkan — retrieval dari seluruh collection")

    # Muat semua baris input
    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                profil_info = extract_profil_from_messages(obj.get("messages", []))
                rows.append(profil_info)
            except json.JSONDecodeError as e:
                logger.warning(
                    "⚠️ Baris %d tidak valid JSON, dilewati: %s", line_num, e
                )

    logger.info("📂 Input: %s (%d profil warga)", input_path, len(rows))
    logger.info("📝 Output: %s", output_path)
    logger.info("🏷️ Reranker model: %s", args.reranker_model)

    # Inisialisasi retriever (sekali saja — model embedding + reranker dimuat 1x)
    logger.info("\n⏳ Memuat PolicyRetriever...")
    retriever = PolicyRetriever(reranker_model_name=args.reranker_model)

    # Proses batch
    t_start = time.time()
    success = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, profil_info in enumerate(rows, 1):
            logger.info("\n[%d/%d]", idx, len(rows))
            try:
                result = retrieve_for_profil(
                    retriever,
                    profil_info,
                    allowed_sources=allowed_sources,
                    top_k=args.top_k,
                    top_n=args.top_n,
                    score_threshold=args.score_threshold,
                    multi_query=not args.single_query,
                )
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                success += 1
            except Exception as e:
                logger.error(
                    "❌ Gagal memproses warga %s: %s", profil_info.get("nik", "?"), e
                )
                failed += 1

    elapsed = time.time() - t_start

    logger.info("\n" + "=" * 60)
    logger.info("✅ Batch selesai!")
    logger.info("   Total warga diproses: %d", len(rows))
    logger.info("   Berhasil            : %d", success)
    logger.info("   Gagal               : %d", failed)
    logger.info(
        "   Waktu total         : %.1f detik (%.1f detik/warga)",
        elapsed,
        elapsed / max(len(rows), 1),
    )
    logger.info("   Output disimpan ke  : %s", output_path.resolve())


if __name__ == "__main__":
    main()
