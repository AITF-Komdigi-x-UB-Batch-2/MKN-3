from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from config import CHUNKED_DIR as CONFIG_CHUNKED_DIR
except Exception:
    CONFIG_CHUNKED_DIR = "chunked_data"


CHUNKED_DIR = Path(CONFIG_CHUNKED_DIR)
PROCESSED_DIR = Path("processed_data")
AUTO_JSONL_PATH = CHUNKED_DIR / "auto_chunks.jsonl"

NORMALIZED_SUFFIX = "_normalized.jsonl"
SHORT_CHUNK_LIMIT = 100
MEDIUM_CHUNK_LIMIT = 500

CORE_METADATA_FIELDS = ("sumber", "page_number")
REQUIRED_CLEAN_METADATA_FIELDS = (
    "nama_bansos",
    "tipe_konten",
    "tipe_konten_primer",
    "retrieval_priority",
)
VALID_RETRIEVAL_PRIORITIES = {"normal", "low"}
MAIN_JUKNIS_SOURCES = {
    "Juklak ASPD Tahun 202620260225_12303533_01.pdf",
    "JUKNIS KEMISKINAN EKSTREM (13-1-2025)-1 (1) (2).pdf",
    "JUKNIS PKH PLUS 2026.pdf",
    "PETUNJUK TEKNIS KIP KPM JAWARA.pdf",
    "Petunjuk Teknis KIP PPKS Jawara 2026.pdf",
    "PETUNJUK TEKNIS KIP PUTRI JAWARA.pdf",
}


@dataclass
class JsonlReadResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    invalid_json_lines: int = 0
    invalid_schema_lines: int = 0


@dataclass
class QualityStats:
    name: str
    filepath: str
    total_chunks: int
    total_characters: int
    avg_length: float
    median_length: float
    min_length: int
    max_length: int
    avg_metadata_keys: float
    empty_chunks: int
    short_chunks: int
    medium_chunks: int
    long_chunks: int
    invalid_json_lines: int
    invalid_schema_lines: int
    duplicate_texts: int
    missing_core_metadata: dict[str, int]
    missing_clean_metadata: dict[str, int]
    invalid_tipe_konten: int
    invalid_retrieval_priority: int
    source_counts: Counter[str]
    priority_counts: Counter[str]
    content_type_counts: Counter[str]
    quality_flag_counts: Counter[str]

    @property
    def warn_count(self) -> int:
        warnings = 0
        warnings += int(self.invalid_json_lines > 0)
        warnings += int(self.invalid_schema_lines > 0)
        warnings += int(self.empty_chunks > 0)
        warnings += int(self.short_ratio > 0.30)
        warnings += int(self.duplicate_texts > 0)
        warnings += int(any(self.missing_core_metadata.values()))
        warnings += int(any(self.missing_clean_metadata.values()))
        warnings += int(self.invalid_tipe_konten > 0)
        warnings += int(self.invalid_retrieval_priority > 0)
        return warnings

    @property
    def short_ratio(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.short_chunks / self.total_chunks


def configure_utf8_stdio() -> None:
    """Make Windows console output safer when UTF-8 is available."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def convert_processed_to_jsonl() -> None:
    """
    Convert legacy processed_data/*.json into chunked_data/auto_chunks.jsonl.

    This is intentionally opt-in because a quality check should not rewrite data
    during normal use.
    """
    if not PROCESSED_DIR.exists():
        print(f"[INFO] Folder legacy tidak ditemukan: {PROCESSED_DIR}")
        return

    CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(PROCESSED_DIR.glob("*.json"))

    print(f"[INFO] Mengonversi {len(json_files)} file legacy ke {AUTO_JSONL_PATH}")
    total_chunks = 0

    with AUTO_JSONL_PATH.open("w", encoding="utf-8") as out_f:
        for filepath in json_files:
            try:
                with filepath.open("r", encoding="utf-8") as in_f:
                    data = json.load(in_f)
            except Exception as exc:
                print(f"[WARN] Gagal membaca {filepath}: {exc}")
                continue

            filename = data.get("filename", filepath.name)
            doc_metadata = data.get("metadata", {})
            kategori = data.get("kategori", "")

            for item in data.get("dokumen_terstruktur", []):
                text = (item.get("text") or item.get("isi") or "").strip()
                if not text:
                    continue

                chunk_metadata = {
                    "filename": filename,
                    "kategori": kategori,
                    "bab": item.get("bab", ""),
                    "pasal": item.get("pasal", ""),
                    **doc_metadata,
                }
                out_f.write(
                    json.dumps(
                        {"text": text, "metadata": chunk_metadata},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total_chunks += 1

    print(f"[OK] Berhasil menulis {total_chunks} chunk ke {AUTO_JSONL_PATH}")


def select_jsonl_files(mode: str) -> list[Path]:
    jsonl_files = sorted(CHUNKED_DIR.glob("*.jsonl"))
    if mode == "all":
        return jsonl_files

    normalized_names = {
        path.name for path in jsonl_files if path.name.endswith(NORMALIZED_SUFFIX)
    }
    selected: list[Path] = []

    for path in jsonl_files:
        if not path.name.endswith(NORMALIZED_SUFFIX):
            expected_normalized = path.name.replace(".jsonl", NORMALIZED_SUFFIX)
            if expected_normalized in normalized_names:
                continue
        selected.append(path)

    return selected


def read_jsonl(filepath: Path) -> JsonlReadResult:
    result = JsonlReadResult()

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                result.invalid_json_lines += 1
                continue

            if not isinstance(row, dict):
                result.invalid_schema_lines += 1
                continue

            if not isinstance(row.get("text"), str) or not isinstance(
                row.get("metadata"), dict
            ):
                result.invalid_schema_lines += 1
                continue

            result.rows.append(row)

    return result


def needs_clean_metadata(metadata: dict[str, Any]) -> bool:
    return metadata.get("sumber") in MAIN_JUKNIS_SOURCES or bool(
        metadata.get("nama_bansos")
    )


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def analyze_chunks(read_result: JsonlReadResult, filepath: Path) -> QualityStats | None:
    chunks = read_result.rows
    if not chunks and read_result.invalid_json_lines == 0:
        return None

    chunk_lengths = [len(row.get("text", "")) for row in chunks]
    total_chunks = len(chunks)
    metadata_keys = [len(row.get("metadata", {}).keys()) for row in chunks]
    text_counts = Counter(row.get("text", "") for row in chunks)

    missing_core_metadata = {field_name: 0 for field_name in CORE_METADATA_FIELDS}
    missing_clean_metadata = {
        field_name: 0 for field_name in REQUIRED_CLEAN_METADATA_FIELDS
    }
    source_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    content_type_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    invalid_tipe_konten = 0
    invalid_retrieval_priority = 0

    for row in chunks:
        metadata = row.get("metadata") or {}

        for field_name in CORE_METADATA_FIELDS:
            if metadata.get(field_name) in (None, "", []):
                missing_core_metadata[field_name] += 1

        source = metadata.get("sumber") or metadata.get("filename") or "(unknown)"
        source_counts[str(source)] += 1

        priority = metadata.get("retrieval_priority")
        if priority:
            priority_counts[str(priority)] += 1

        for tipe in as_list(metadata.get("tipe_konten")):
            content_type_counts[str(tipe)] += 1

        for flag in as_list(metadata.get("quality_flags")):
            quality_flag_counts[str(flag)] += 1

        if not needs_clean_metadata(metadata):
            continue

        for field_name in REQUIRED_CLEAN_METADATA_FIELDS:
            if metadata.get(field_name) in (None, "", []):
                missing_clean_metadata[field_name] += 1

        tipe_konten = metadata.get("tipe_konten")
        if not isinstance(tipe_konten, list) or not tipe_konten:
            invalid_tipe_konten += 1

        if metadata.get("retrieval_priority") not in VALID_RETRIEVAL_PRIORITIES:
            invalid_retrieval_priority += 1

    return QualityStats(
        name=filepath.name,
        filepath=str(filepath),
        total_chunks=total_chunks,
        total_characters=sum(chunk_lengths),
        avg_length=statistics.mean(chunk_lengths) if total_chunks else 0,
        median_length=statistics.median(chunk_lengths) if total_chunks else 0,
        min_length=min(chunk_lengths) if total_chunks else 0,
        max_length=max(chunk_lengths) if total_chunks else 0,
        avg_metadata_keys=(sum(metadata_keys) / total_chunks if total_chunks else 0),
        empty_chunks=sum(1 for length in chunk_lengths if length == 0),
        short_chunks=sum(1 for length in chunk_lengths if length < SHORT_CHUNK_LIMIT),
        medium_chunks=sum(
            1 for length in chunk_lengths if SHORT_CHUNK_LIMIT <= length <= MEDIUM_CHUNK_LIMIT
        ),
        long_chunks=sum(1 for length in chunk_lengths if length > MEDIUM_CHUNK_LIMIT),
        invalid_json_lines=read_result.invalid_json_lines,
        invalid_schema_lines=read_result.invalid_schema_lines,
        duplicate_texts=sum(count - 1 for count in text_counts.values() if count > 1),
        missing_core_metadata=missing_core_metadata,
        missing_clean_metadata=missing_clean_metadata,
        invalid_tipe_konten=invalid_tipe_konten,
        invalid_retrieval_priority=invalid_retrieval_priority,
        source_counts=source_counts,
        priority_counts=priority_counts,
        content_type_counts=content_type_counts,
        quality_flag_counts=quality_flag_counts,
    )


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total * 100


def format_counter(counter: Counter[str], limit: int = 5) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def print_stats(stats: QualityStats) -> None:
    print(f"\n=== Statistik Data: {stats.name} ===")
    print(f"Total chunk             : {stats.total_chunks:,}")
    print(f"Total karakter          : {stats.total_characters:,}")
    print(f"Rata-rata panjang       : {stats.avg_length:.2f} karakter")
    print(f"Median panjang          : {stats.median_length:.2f} karakter")
    print(f"Panjang min - max       : {stats.min_length} - {stats.max_length} karakter")
    print(f"Rata-rata metadata      : {stats.avg_metadata_keys:.2f} fields")
    print(f"Chunk kosong            : {stats.empty_chunks}")
    print(
        f"Chunk pendek (<{SHORT_CHUNK_LIMIT})    : "
        f"{stats.short_chunks} ({pct(stats.short_chunks, stats.total_chunks):.1f}%)"
    )
    print(
        f"Chunk sedang ({SHORT_CHUNK_LIMIT}-{MEDIUM_CHUNK_LIMIT}): "
        f"{stats.medium_chunks} ({pct(stats.medium_chunks, stats.total_chunks):.1f}%)"
    )
    print(
        f"Chunk panjang (>{MEDIUM_CHUNK_LIMIT})   : "
        f"{stats.long_chunks} ({pct(stats.long_chunks, stats.total_chunks):.1f}%)"
    )
    print(f"Invalid JSON line       : {stats.invalid_json_lines}")
    print(f"Invalid schema line     : {stats.invalid_schema_lines}")
    print(f"Duplikat teks           : {stats.duplicate_texts}")
    print(f"Sumber teratas          : {format_counter(stats.source_counts)}")
    print(f"Retrieval priority      : {format_counter(stats.priority_counts)}")
    print(f"Tipe konten             : {format_counter(stats.content_type_counts)}")
    print(f"Quality flags           : {format_counter(stats.quality_flag_counts)}")

    warnings = []
    if stats.empty_chunks:
        warnings.append(f"{stats.empty_chunks} chunk kosong")
    if stats.short_ratio > 0.30:
        warnings.append(
            f"{pct(stats.short_chunks, stats.total_chunks):.1f}% chunk pendek"
        )
    if stats.invalid_json_lines:
        warnings.append(f"{stats.invalid_json_lines} baris JSON rusak")
    if stats.invalid_schema_lines:
        warnings.append(f"{stats.invalid_schema_lines} baris schema tidak valid")
    if stats.duplicate_texts:
        warnings.append(f"{stats.duplicate_texts} duplikat teks")

    for field_name, count in stats.missing_core_metadata.items():
        if count:
            warnings.append(f"metadata '{field_name}' kosong di {count} chunk")

    for field_name, count in stats.missing_clean_metadata.items():
        if count:
            warnings.append(f"metadata clean '{field_name}' kosong di {count} chunk")

    if stats.invalid_tipe_konten:
        warnings.append(f"tipe_konten invalid di {stats.invalid_tipe_konten} chunk")
    if stats.invalid_retrieval_priority:
        warnings.append(
            f"retrieval_priority invalid di {stats.invalid_retrieval_priority} chunk"
        )

    if warnings:
        print("Catatan kualitas        :")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Catatan kualitas        : OK")


def print_comparative_summary(all_stats: list[QualityStats]) -> None:
    if not all_stats:
        return

    total_chunks = sum(stats.total_chunks for stats in all_stats)
    total_characters = sum(stats.total_characters for stats in all_stats)
    total_warnings = sum(stats.warn_count for stats in all_stats)

    print("\n=== Ringkasan Quality Check ===")
    print(f"File dicek              : {len(all_stats)}")
    print(f"Total chunk valid       : {total_chunks:,}")
    print(f"Total karakter          : {total_characters:,}")
    print(f"Total indikator warning : {total_warnings}")

    sorted_by_length = sorted(all_stats, key=lambda item: item.avg_length, reverse=True)
    sorted_by_metadata = sorted(
        all_stats, key=lambda item: item.avg_metadata_keys, reverse=True
    )

    print(
        "Dataset paling deskriptif: "
        f"{sorted_by_length[0].name} ({sorted_by_length[0].avg_length:.1f} char)"
    )
    print(
        "Dataset metadata terkaya : "
        f"{sorted_by_metadata[0].name} ({sorted_by_metadata[0].avg_metadata_keys:.1f} fields)"
    )

    risky_files = [stats for stats in all_stats if stats.warn_count > 0]
    if risky_files:
        print("\nFile yang perlu dicek lagi:")
        for stats in sorted(risky_files, key=lambda item: item.warn_count, reverse=True):
            print(f"  - {stats.name}: {stats.warn_count} indikator warning")
    else:
        print("\nSemua file lolos cek dasar.")


def build_report(all_stats: list[QualityStats], selected_files: list[Path]) -> dict[str, Any]:
    return {
        "chunked_dir": str(CHUNKED_DIR),
        "files_selected": [path.name for path in selected_files],
        "files": [
            {
                "name": stats.name,
                "filepath": stats.filepath,
                "total_chunks": stats.total_chunks,
                "total_characters": stats.total_characters,
                "avg_length": stats.avg_length,
                "median_length": stats.median_length,
                "min_length": stats.min_length,
                "max_length": stats.max_length,
                "avg_metadata_keys": stats.avg_metadata_keys,
                "empty_chunks": stats.empty_chunks,
                "short_chunks": stats.short_chunks,
                "medium_chunks": stats.medium_chunks,
                "long_chunks": stats.long_chunks,
                "invalid_json_lines": stats.invalid_json_lines,
                "invalid_schema_lines": stats.invalid_schema_lines,
                "duplicate_texts": stats.duplicate_texts,
                "missing_core_metadata": stats.missing_core_metadata,
                "missing_clean_metadata": stats.missing_clean_metadata,
                "invalid_tipe_konten": stats.invalid_tipe_konten,
                "invalid_retrieval_priority": stats.invalid_retrieval_priority,
                "source_counts": dict(stats.source_counts),
                "priority_counts": dict(stats.priority_counts),
                "content_type_counts": dict(stats.content_type_counts),
                "quality_flag_counts": dict(stats.quality_flag_counts),
                "warning_indicators": stats.warn_count,
            }
            for stats in all_stats
        ],
    }


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Report JSON ditulis ke {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quality check untuk file JSONL di chunked_data."
    )
    parser.add_argument(
        "--mode",
        choices=("ingest", "all"),
        default="ingest",
        help=(
            "ingest: cek file yang akan dipakai 04_embed_and_ingest_v2.py; "
            "all: cek semua file .jsonl."
        ),
    )
    parser.add_argument(
        "--convert-legacy",
        action="store_true",
        help="Konversi processed_data/*.json ke chunked_data/auto_chunks.jsonl.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Opsional: path output report JSON, contoh evaluation/results/quality_report.json.",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()

    if args.convert_legacy:
        convert_processed_to_jsonl()

    if not CHUNKED_DIR.exists():
        print(f"[ERROR] Folder tidak ditemukan: {CHUNKED_DIR}")
        return

    selected_files = select_jsonl_files(args.mode)
    if not selected_files:
        print(f"[WARN] Tidak ada file .jsonl ditemukan di {CHUNKED_DIR}")
        return

    print("=== Quality Check JSONL ===")
    print(f"Folder data : {CHUNKED_DIR}")
    print(f"Mode        : {args.mode}")
    print(f"File dicek  : {len(selected_files)}")
    if args.mode == "ingest":
        print("Catatan     : file asli dilewati bila versi _normalized tersedia")

    all_stats: list[QualityStats] = []
    for filepath in selected_files:
        read_result = read_jsonl(filepath)
        stats = analyze_chunks(read_result, filepath)
        if stats:
            all_stats.append(stats)
            print_stats(stats)

    print_comparative_summary(all_stats)

    if args.report:
        write_report(args.report, build_report(all_stats, selected_files))


if __name__ == "__main__":
    main()
