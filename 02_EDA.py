from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import CHUNKED_DIR


# ============================================================
# Stage 02: Exploratory Data Analysis for JSONL RAG chunks
#
# Default mode mirrors 04_embed_and_ingest_v2.py:
# if a *_normalized.jsonl file exists, the raw file is skipped.
# ============================================================


CHUNKED_DIR_PATH = Path(CHUNKED_DIR)
OUTPUT_DIR = Path(__file__).resolve().parent / "eda_output"

NORMALIZED_SUFFIX = "_normalized.jsonl"
SHORT_CHUNK_LIMIT = 100
MEDIUM_CHUNK_LIMIT = 500
LONG_CHUNK_LIMIT = 2000

TEXT_COL = "text"
SOURCE_COL = "sumber"
PAGE_COL = "page_number"
CATEGORY_COL = "kategori"
BAB_COL = "bab"
PASAL_COL = "pasal"
AYAT_COL = "ayat"
TITLE_COL = "judul_halaman"
PROGRAM_COL = "nama_bansos"
CONTENT_TYPE_COL = "tipe_konten"
PRIMARY_CONTENT_TYPE_COL = "tipe_konten_primer"
PRIORITY_COL = "retrieval_priority"
QUALITY_FLAGS_COL = "quality_flags"

METADATA_COLUMNS = (
    SOURCE_COL,
    PAGE_COL,
    CATEGORY_COL,
    BAB_COL,
    PASAL_COL,
    AYAT_COL,
    TITLE_COL,
    PROGRAM_COL,
    CONTENT_TYPE_COL,
    PRIMARY_CONTENT_TYPE_COL,
    PRIORITY_COL,
    QUALITY_FLAGS_COL,
)

REQUIRED_CLEAN_METADATA_FIELDS = (
    PROGRAM_COL,
    CONTENT_TYPE_COL,
    PRIMARY_CONTENT_TYPE_COL,
    PRIORITY_COL,
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    return str(value)


def safe_len(value: Any) -> int:
    return len(safe_text(value).strip())


def as_items(value: Any) -> list[str]:
    if not is_filled(value):
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


def truncate(value: Any, max_len: int = 60) -> str:
    text = safe_text(value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def select_jsonl_files(mode: str) -> list[Path]:
    jsonl_files = sorted(CHUNKED_DIR_PATH.glob("*.jsonl"))
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


def read_jsonl_flat(path: Path) -> tuple[pd.DataFrame, int, int]:
    rows: list[dict[str, Any]] = []
    invalid_json_lines = 0
    invalid_schema_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue

            if not isinstance(obj, dict):
                invalid_schema_lines += 1
                continue

            text = obj.get(TEXT_COL)
            metadata = obj.get("metadata")
            if not isinstance(text, str) or not isinstance(metadata, dict):
                logger.debug("Invalid schema at %s:%s", path.name, line_num)
                invalid_schema_lines += 1
                continue

            row = {
                "_source_file": path.name,
                TEXT_COL: text,
                "_text_len": len(text.strip()),
            }
            row.update(metadata)
            rows.append(row)

    return pd.DataFrame(rows), invalid_json_lines, invalid_schema_lines


def read_all(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summary_rows = []

    for path in files:
        df, invalid_json, invalid_schema = read_jsonl_flat(path)
        if not df.empty:
            frames.append(df)

        text_lens = df["_text_len"] if "_text_len" in df.columns else pd.Series(dtype=int)
        valid_lens = text_lens[text_lens > 0]
        source_count = count_values(df, SOURCE_COL)
        category_count = count_values(df, CATEGORY_COL)
        program_count = count_values(df, PROGRAM_COL)

        summary_rows.append(
            {
                "file": path.name,
                "chunks": int(len(df)),
                "invalid_json": invalid_json,
                "invalid_schema": invalid_schema,
                "empty_chunks": int((text_lens == 0).sum()) if not df.empty else 0,
                "short_chunks": int((text_lens < SHORT_CHUNK_LIMIT).sum())
                if not df.empty
                else 0,
                "avg_len": round(float(valid_lens.mean()), 1)
                if not valid_lens.empty
                else 0,
                "median_len": round(float(valid_lens.median()), 1)
                if not valid_lens.empty
                else 0,
                "min_len": int(valid_lens.min()) if not valid_lens.empty else 0,
                "max_len": int(valid_lens.max()) if not valid_lens.empty else 0,
                "sources": len(source_count),
                "top_source": truncate(source_count.most_common(1)[0][0])
                if source_count
                else "-",
                "top_category": truncate(category_count.most_common(1)[0][0])
                if category_count
                else "-",
                "top_program": truncate(program_count.most_common(1)[0][0])
                if program_count
                else "-",
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return combined, summary


def count_values(df: pd.DataFrame, column: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if column not in df.columns:
        return counts

    for value in df[column]:
        for item in as_items(value):
            counts[item] += 1

    return counts


def value_counts_series(df: pd.DataFrame, column: str, fallback: str | None = None) -> pd.Series:
    values: list[str] = []
    if column in df.columns:
        for value in df[column]:
            values.extend(as_items(value))

    if not values and fallback and fallback in df.columns:
        values = [safe_text(value) for value in df[fallback] if is_filled(value)]

    if not values:
        return pd.Series(dtype=int)

    return pd.Series(Counter(values)).sort_values(ascending=False)


def has_any_structure(row: pd.Series) -> bool:
    return any(
        is_filled(row.get(column))
        for column in (BAB_COL, PASAL_COL, AYAT_COL, TITLE_COL, CATEGORY_COL, PROGRAM_COL)
    )


def needs_clean_metadata(row: pd.Series) -> bool:
    return row.get(SOURCE_COL) in MAIN_JUKNIS_SOURCES or is_filled(row.get(PROGRAM_COL))


def build_stats(combined: pd.DataFrame, file_summary: pd.DataFrame) -> dict[str, Any]:
    if combined.empty:
        return {}

    text_lens = combined["_text_len"]
    valid = combined[text_lens > 0].copy()
    valid_lens = valid["_text_len"]

    completeness = {}
    for column in (TEXT_COL, *METADATA_COLUMNS):
        if column not in combined.columns:
            completeness[column] = 0.0
            continue
        filled = sum(1 for value in combined[column] if is_filled(value))
        completeness[column] = round(filled / len(combined) * 100, 1)

    main_program_rows = valid[valid.apply(needs_clean_metadata, axis=1)]

    missing_clean_metadata: dict[str, int] = {}
    for field in REQUIRED_CLEAN_METADATA_FIELDS:
        if field not in main_program_rows.columns:
            missing_clean_metadata[field] = int(len(main_program_rows))
            continue
        missing_clean_metadata[field] = int(
            sum(1 for value in main_program_rows[field] if not is_filled(value))
        )

    invalid_tipe_konten = 0
    if CONTENT_TYPE_COL in main_program_rows.columns:
        invalid_tipe_konten = int(
            sum(
                1
                for value in main_program_rows[CONTENT_TYPE_COL]
                if not isinstance(value, list) or not value
            )
        )
    else:
        invalid_tipe_konten = int(len(main_program_rows))

    invalid_priority = 0
    if PRIORITY_COL in main_program_rows.columns:
        invalid_priority = int(
            sum(
                1
                for value in main_program_rows[PRIORITY_COL]
                if value not in VALID_RETRIEVAL_PRIORITIES
            )
        )
    else:
        invalid_priority = int(len(main_program_rows))

    duplicate_texts = int(valid.duplicated(subset=[TEXT_COL]).sum())

    structure_counts = {
        "bab": int(sum(1 for value in valid.get(BAB_COL, []) if is_filled(value)))
        if BAB_COL in valid.columns
        else 0,
        "pasal": int(sum(1 for value in valid.get(PASAL_COL, []) if is_filled(value)))
        if PASAL_COL in valid.columns
        else 0,
        "ayat": int(sum(1 for value in valid.get(AYAT_COL, []) if is_filled(value)))
        if AYAT_COL in valid.columns
        else 0,
        "judul_halaman": int(
            sum(1 for value in valid.get(TITLE_COL, []) if is_filled(value))
        )
        if TITLE_COL in valid.columns
        else 0,
        "any_context_structure": int(valid.apply(has_any_structure, axis=1).sum()),
    }

    invalid_json_total = int(file_summary["invalid_json"].sum()) if not file_summary.empty else 0
    invalid_schema_total = (
        int(file_summary["invalid_schema"].sum()) if not file_summary.empty else 0
    )

    return {
        "combined": combined,
        "valid": valid,
        "file_summary": file_summary,
        "total_files": int(len(file_summary)),
        "total_chunks": int(len(combined)),
        "valid_chunks": int(len(valid)),
        "empty_chunks": int((combined["_text_len"] == 0).sum()),
        "short_chunks": int((valid_lens < SHORT_CHUNK_LIMIT).sum()),
        "medium_chunks": int(
            ((valid_lens >= SHORT_CHUNK_LIMIT) & (valid_lens <= MEDIUM_CHUNK_LIMIT)).sum()
        ),
        "long_chunks": int((valid_lens > MEDIUM_CHUNK_LIMIT).sum()),
        "very_long_chunks": int((valid_lens > LONG_CHUNK_LIMIT).sum()),
        "avg_len": float(valid_lens.mean()) if not valid_lens.empty else 0.0,
        "median_len": float(valid_lens.median()) if not valid_lens.empty else 0.0,
        "min_len": int(valid_lens.min()) if not valid_lens.empty else 0,
        "max_len": int(valid_lens.max()) if not valid_lens.empty else 0,
        "std_len": float(valid_lens.std()) if len(valid_lens) > 1 else 0.0,
        "completeness": completeness,
        "source_counts": value_counts_series(valid, SOURCE_COL, fallback="_source_file"),
        "file_counts": valid.groupby("_source_file").size().sort_values(ascending=False),
        "category_counts": value_counts_series(valid, CATEGORY_COL),
        "program_counts": value_counts_series(valid, PROGRAM_COL),
        "content_type_counts": value_counts_series(valid, CONTENT_TYPE_COL),
        "primary_type_counts": value_counts_series(valid, PRIMARY_CONTENT_TYPE_COL),
        "priority_counts": value_counts_series(valid, PRIORITY_COL),
        "quality_flag_counts": value_counts_series(valid, QUALITY_FLAGS_COL),
        "structure_counts": structure_counts,
        "main_program_chunks": int(len(main_program_rows)),
        "missing_clean_metadata": missing_clean_metadata,
        "invalid_tipe_konten": invalid_tipe_konten,
        "invalid_priority": invalid_priority,
        "duplicate_texts": duplicate_texts,
        "invalid_json_total": invalid_json_total,
        "invalid_schema_total": invalid_schema_total,
    }


def render_file_summary(file_summary: pd.DataFrame) -> None:
    table = Table(
        title="Ringkasan per File JSONL",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("No", justify="right", width=3)
    table.add_column("File", max_width=38)
    table.add_column("Chunk", justify="right")
    table.add_column("Avg", justify="right")
    table.add_column("Med", justify="right")
    table.add_column("Min-Max", justify="right")
    table.add_column("Short", justify="right")
    table.add_column("Program/Kategori", max_width=32)

    for idx, row in file_summary.iterrows():
        program_or_category = row["top_program"]
        if program_or_category == "-":
            program_or_category = row["top_category"]
        table.add_row(
            str(idx + 1),
            row["file"],
            f"{int(row['chunks']):,}",
            f"{float(row['avg_len']):,.1f}",
            f"{float(row['median_len']):,.1f}",
            f"{int(row['min_len'])}-{int(row['max_len'])}",
            str(int(row["short_chunks"])),
            program_or_category,
        )

    console.print(table)


def render_global_stats(stats: dict[str, Any]) -> None:
    body = (
        f"Total file JSONL     : {stats['total_files']}\n"
        f"Total chunk          : {stats['total_chunks']:,}\n"
        f"Chunk valid          : {stats['valid_chunks']:,}\n"
        f"Chunk kosong         : {stats['empty_chunks']:,}\n"
        f"Chunk pendek <100    : {stats['short_chunks']:,}\n"
        f"Chunk sedang 100-500 : {stats['medium_chunks']:,}\n"
        f"Chunk panjang >500   : {stats['long_chunks']:,}\n"
        f"Chunk >2000          : {stats['very_long_chunks']:,}\n\n"
        f"Rata-rata panjang    : {stats['avg_len']:,.1f} karakter\n"
        f"Median panjang       : {stats['median_len']:,.1f} karakter\n"
        f"Min - Max            : {stats['min_len']:,} - {stats['max_len']:,} karakter\n"
        f"Std dev              : {stats['std_len']:,.1f} karakter"
    )
    console.print(Panel(body, title="Statistik Global", box=box.DOUBLE, border_style="cyan"))

    completeness_table = Table(
        title="Kelengkapan Field Utama",
        box=box.SIMPLE_HEAVY,
        header_style="bold green",
    )
    completeness_table.add_column("Field")
    completeness_table.add_column("% Terisi", justify="right")
    completeness_table.add_column("Status", justify="center")

    for column, percent in stats["completeness"].items():
        if percent >= 95:
            status = "OK"
            style = "green"
        elif percent >= 70:
            status = "WARN"
            style = "yellow"
        else:
            status = "LOW"
            style = "red"
        completeness_table.add_row(column, f"[{style}]{percent:.1f}%[/{style}]", status)

    console.print(completeness_table)

    dist_table = Table(
        title="Distribusi Metadata RAG",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
    )
    dist_table.add_column("Dimensi")
    dist_table.add_column("Top Value")
    dist_table.add_column("Count", justify="right")

    dimensions = [
        ("Sumber", stats["source_counts"]),
        ("Program", stats["program_counts"]),
        ("Tipe konten", stats["content_type_counts"]),
        ("Priority", stats["priority_counts"]),
        ("Quality flag", stats["quality_flag_counts"]),
    ]
    for name, series in dimensions:
        if series.empty:
            dist_table.add_row(name, "-", "0")
            continue
        top_key = str(series.index[0])
        top_count = int(series.iloc[0])
        dist_table.add_row(name, truncate(top_key, 70), f"{top_count:,}")

    console.print(dist_table)


def render_quality_checks(stats: dict[str, Any]) -> None:
    checks: list[tuple[str, str]] = []
    total = max(stats["total_chunks"], 1)
    valid = max(stats["valid_chunks"], 1)

    if stats["invalid_json_total"] == 0 and stats["invalid_schema_total"] == 0:
        checks.append(("OK", "Semua baris JSONL valid dan sesuai schema text/metadata."))
    else:
        checks.append(
            (
                "WARN",
                f"Ada {stats['invalid_json_total']} JSON rusak dan "
                f"{stats['invalid_schema_total']} schema invalid.",
            )
        )

    if stats["empty_chunks"] == 0:
        checks.append(("OK", "Tidak ada chunk kosong."))
    else:
        checks.append(
            (
                "WARN",
                f"{stats['empty_chunks']} chunk kosong "
                f"({stats['empty_chunks'] / total * 100:.1f}%).",
            )
        )

    short_ratio = stats["short_chunks"] / valid
    if short_ratio <= 0.30:
        checks.append(("OK", f"Chunk pendek masih aman ({short_ratio * 100:.1f}%)."))
    else:
        checks.append(
            (
                "WARN",
                f"Chunk pendek terlalu banyak ({short_ratio * 100:.1f}%).",
            )
        )

    source_pct = stats["completeness"].get(SOURCE_COL, 0.0)
    page_pct = stats["completeness"].get(PAGE_COL, 0.0)
    if source_pct >= 99:
        checks.append(("OK", f"Metadata sumber lengkap ({source_pct:.1f}%)."))
    else:
        checks.append(("WARN", f"Metadata sumber hanya {source_pct:.1f}%."))

    if page_pct >= 90:
        checks.append(("OK", f"Metadata page_number memadai ({page_pct:.1f}%)."))
    else:
        checks.append(("INFO", f"Metadata page_number terisi {page_pct:.1f}%."))

    if stats["duplicate_texts"] == 0:
        checks.append(("OK", "Tidak ada duplikat teks antar chunk."))
    else:
        checks.append(("WARN", f"Ada {stats['duplicate_texts']} duplikat teks."))

    missing_clean = {
        field: count
        for field, count in stats["missing_clean_metadata"].items()
        if count > 0
    }
    if not missing_clean and stats["invalid_tipe_konten"] == 0 and stats["invalid_priority"] == 0:
        checks.append(
            (
                "OK",
                f"Metadata clean untuk {stats['main_program_chunks']} chunk Juknis utama lengkap.",
            )
        )
    else:
        detail = ", ".join(f"{field}={count}" for field, count in missing_clean.items())
        checks.append(
            (
                "WARN",
                "Metadata clean Juknis belum lengkap"
                + (f": {detail}" if detail else ".")
                + f" invalid_tipe_konten={stats['invalid_tipe_konten']},"
                + f" invalid_priority={stats['invalid_priority']}.",
            )
        )

    context_ratio = stats["structure_counts"]["any_context_structure"] / valid
    if context_ratio >= 0.90:
        checks.append(("OK", f"Konteks struktur/judul/kategori tersedia {context_ratio * 100:.1f}%."))
    else:
        checks.append(("INFO", f"Konteks struktur/judul/kategori tersedia {context_ratio * 100:.1f}%."))

    table = Table(
        title="Quality Notes untuk RAG",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Status", width=8)
    table.add_column("Catatan")
    for status, message in checks:
        style = "green" if status == "OK" else ("yellow" if status == "INFO" else "red")
        table.add_row(f"[{style}]{status}[/{style}]", message)

    console.print(table)


def save_dataframe_outputs(file_summary: pd.DataFrame, stats: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_summary.to_csv(output_dir / "document_summary.csv", index=False, encoding="utf-8")

    metadata_rows = []
    for field, percent in stats["completeness"].items():
        metadata_rows.append({"field": field, "filled_percent": percent})
    pd.DataFrame(metadata_rows).to_csv(
        output_dir / "metadata_completeness.csv", index=False, encoding="utf-8"
    )


def apply_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#2f3a4a",
            "axes.labelcolor": "#1f2937",
            "text.color": "#1f2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#d1d5db",
            "font.size": 10,
        }
    )


def save_bar_chart(series: pd.Series, title: str, path: Path, top_n: int = 15) -> None:
    if series.empty:
        return

    data = series.head(top_n).sort_values(ascending=True)
    labels = [truncate(label, 48) for label in data.index]
    height = max(4.5, len(data) * 0.45)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, data.values, color="#2563eb")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Jumlah chunk")
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    for idx, value in enumerate(data.values):
        ax.text(value + max(data.values) * 0.01, idx, f"{int(value):,}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_charts(stats: dict[str, Any], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_chart_style()
    chart_files: list[str] = []

    valid_lens = stats["valid"]["_text_len"]
    if not valid_lens.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(valid_lens, bins=40, color="#2563eb", edgecolor="#eff6ff", alpha=0.9)
        ax.axvline(valid_lens.median(), color="#dc2626", linestyle="--", label="Median")
        ax.axvline(valid_lens.mean(), color="#f59e0b", linestyle="--", label="Mean")
        ax.set_title("Distribusi Panjang Teks per Chunk", fontsize=13, fontweight="bold")
        ax.set_xlabel("Jumlah karakter")
        ax.set_ylabel("Frekuensi")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.45)
        fig.tight_layout()
        path = output_dir / "01_text_length_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        chart_files.append(path.name)

    chart_specs = [
        ("02_chunks_per_file.png", "Jumlah Chunk per File JSONL", stats["file_counts"]),
        ("03_chunks_per_source.png", "Jumlah Chunk per Sumber Dokumen", stats["source_counts"]),
        ("04_program_distribution.png", "Distribusi Program Bansos", stats["program_counts"]),
        ("05_content_type_distribution.png", "Distribusi Tipe Konten", stats["content_type_counts"]),
        ("06_retrieval_priority_distribution.png", "Distribusi Retrieval Priority", stats["priority_counts"]),
        ("07_quality_flags_distribution.png", "Distribusi Quality Flags", stats["quality_flag_counts"]),
    ]

    for filename, title, series in chart_specs:
        if series.empty:
            continue
        save_bar_chart(series, title, output_dir / filename)
        chart_files.append(filename)

    comp = pd.Series(stats["completeness"]).sort_values(ascending=True)
    if not comp.empty:
        fig, ax = plt.subplots(figsize=(10, max(4.5, len(comp) * 0.35)))
        ax.barh(comp.index, comp.values, color="#059669")
        ax.set_title("Kelengkapan Field Utama", fontsize=13, fontweight="bold")
        ax.set_xlabel("% terisi")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle="--", alpha=0.45)
        for idx, value in enumerate(comp.values):
            ax.text(value + 1, idx, f"{value:.1f}%", va="center", fontsize=8)
        fig.tight_layout()
        path = output_dir / "08_metadata_completeness.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        chart_files.append(path.name)

    return chart_files


def generate_html_report(
    file_summary: pd.DataFrame,
    stats: dict[str, Any],
    chart_files: list[str],
    output_dir: Path,
    mode: str,
) -> Path:
    table_rows = []
    for _, row in file_summary.iterrows():
        table_rows.append(
            "<tr>"
            f"<td>{row['file']}</td>"
            f"<td>{int(row['chunks']):,}</td>"
            f"<td>{float(row['avg_len']):,.1f}</td>"
            f"<td>{float(row['median_len']):,.1f}</td>"
            f"<td>{int(row['min_len'])}-{int(row['max_len'])}</td>"
            f"<td>{int(row['short_chunks'])}</td>"
            f"<td>{row['top_program']}</td>"
            "</tr>"
        )

    chart_html = "\n".join(
        f'<section class="chart"><img src="{filename}" alt="{filename}"></section>'
        for filename in chart_files
    )

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <title>EDA Report - RAG Kebijakan Sosial</title>
  <style>
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      background: #f8fafc;
      color: #111827;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px;
    }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #6b7280; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .stat {{
      border: 1px solid #e5e7eb;
      background: #ffffff;
      border-radius: 8px;
      padding: 16px;
    }}
    .value {{
      font-size: 28px;
      font-weight: 700;
      color: #1d4ed8;
    }}
    .label {{ color: #6b7280; margin-top: 4px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      margin: 16px 0 28px;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eff6ff; color: #1e3a8a; }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
      gap: 18px;
    }}
    .chart {{
      border: 1px solid #e5e7eb;
      background: #ffffff;
      border-radius: 8px;
      padding: 12px;
    }}
    img {{ max-width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
<main>
  <h1>EDA Report - RAG Kebijakan Sosial</h1>
  <p class="muted">Mode: {mode}. Default mode mengikuti file target ingest.</p>

  <div class="grid">
    <div class="stat"><div class="value">{stats['total_files']}</div><div class="label">File JSONL</div></div>
    <div class="stat"><div class="value">{stats['valid_chunks']:,}</div><div class="label">Chunk valid</div></div>
    <div class="stat"><div class="value">{stats['avg_len']:,.0f}</div><div class="label">Rata-rata karakter</div></div>
    <div class="stat"><div class="value">{stats['median_len']:,.0f}</div><div class="label">Median karakter</div></div>
    <div class="stat"><div class="value">{stats['short_chunks']:,}</div><div class="label">Chunk pendek</div></div>
    <div class="stat"><div class="value">{stats['duplicate_texts']:,}</div><div class="label">Duplikat teks</div></div>
  </div>

  <h2>Ringkasan per File</h2>
  <table>
    <thead>
      <tr>
        <th>File</th><th>Chunk</th><th>Avg</th><th>Median</th>
        <th>Min-Max</th><th>Short</th><th>Program</th>
      </tr>
    </thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>

  <h2>Visualisasi</h2>
  <div class="charts">
    {chart_html}
  </div>
</main>
</body>
</html>
"""

    report_path = output_dir / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA untuk file JSONL RAG di chunked_data."
    )
    parser.add_argument(
        "--mode",
        choices=("ingest", "all"),
        default="ingest",
        help=(
            "ingest: analisis file yang akan dipakai 04_embed_and_ingest_v2.py; "
            "all: analisis semua file .jsonl termasuk raw/legacy."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Folder output chart, CSV, dan HTML report.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Lewati pembuatan chart PNG dan HTML report.",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()

    if not CHUNKED_DIR_PATH.exists():
        console.print(f"[bold red]Folder JSONL tidak ditemukan: {CHUNKED_DIR_PATH}[/]")
        sys.exit(1)

    files = select_jsonl_files(args.mode)
    if not files:
        console.print(f"[bold red]Tidak ada file JSONL di {CHUNKED_DIR_PATH}[/]")
        sys.exit(1)

    console.print(
        Panel(
            f"Folder data : {CHUNKED_DIR_PATH}\n"
            f"Mode        : {args.mode}\n"
            f"File dicek  : {len(files)}\n"
            + (
                "Catatan     : raw JSONL dilewati bila versi _normalized tersedia"
                if args.mode == "ingest"
                else "Catatan     : semua JSONL dianalisis"
            ),
            title="Stage 02 - EDA JSONL",
            border_style="cyan",
            box=box.DOUBLE,
        )
    )

    combined, file_summary = read_all(files)
    if combined.empty:
        console.print("[bold red]Tidak ada data valid untuk dianalisis.[/]")
        sys.exit(1)

    stats = build_stats(combined, file_summary)
    render_file_summary(file_summary)
    render_global_stats(stats)
    render_quality_checks(stats)

    save_dataframe_outputs(file_summary, stats, args.output_dir)

    if args.no_charts:
        console.print(
            Panel(
                f"CSV summary ditulis ke: {args.output_dir}",
                title="EDA selesai",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        return

    chart_files = create_charts(stats, args.output_dir)
    report_path = generate_html_report(
        file_summary=file_summary,
        stats=stats,
        chart_files=chart_files,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    console.print(
        Panel(
            f"Chart PNG   : {args.output_dir}\n"
            f"CSV summary : {args.output_dir / 'document_summary.csv'}\n"
            f"HTML report : {report_path}",
            title="EDA selesai",
            border_style="green",
            box=box.DOUBLE,
        )
    )


if __name__ == "__main__":
    main()
