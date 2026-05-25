# ============================================================
# 07_semantic_search_ui_v2.py — Semantic Search UI v2 (CSV payload)
# Social Welfare Policy Recommender System (Tim 4)
#
# Menampilkan hasil retrieval + reranking tanpa LLM generation.
# v2: Metadata Bab/Pasal/Ayat/Konteks_Lengkap dari kolom CSV
#     (bukan regex-extracted header_1/header_2/header_3)
# ============================================================

import os
import time
import logging

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box

from config import (
    QDRANT_COLLECTION, EMBED_MODEL_NAME, RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K, RERANK_TOP_N,
    configure_utf8_stdio,
)
configure_utf8_stdio()

from retrieval import PolicyRetriever, RetrievalResult

# ============================================================
# SETUP
# ============================================================

console = Console(soft_wrap=True)


def silence_loggers():
    """Matikan semua INFO logs agar hanya rich UI yang tampil."""
    logging.getLogger().setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)


# ============================================================
# DISPLAY
# ============================================================

def render_header():
    """Header aplikasi."""
    header = Text()
    header.append("  SEMANTIC SEARCH v2", style="bold cyan")
    header.append("  ░▒▓  ", style="dim cyan")
    header.append("Kebijakan Sosial", style="bold white")
    header.append("  ░▒▓  ", style="dim cyan")
    header.append("Tim 4 UB\n", style="bold white")

    info = Text()
    info.append("  ◈ Embedding  ", style="dim")
    info.append(f"{EMBED_MODEL_NAME}\n", style="green")
    info.append("  ◈ Reranker   ", style="dim")
    info.append(f"{RERANKER_MODEL_NAME}\n", style="green")
    info.append("  ◈ Collection ", style="dim")
    info.append(f"{QDRANT_COLLECTION}", style="green")
    info.append(f"  │  Top-K={RETRIEVAL_TOP_K}  Top-N={RERANK_TOP_N}\n", style="dim")
    info.append("\n  Ketik query pencarian. Ketik ", style="dim white")
    info.append("exit", style="bold red")
    info.append(" untuk keluar.", style="dim white")

    content = Text()
    content.append_text(header)
    content.append_text(info)

    console.print(Panel(
        content,
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    ))


def render_query_header(query: str, retrieval_ms: float, result_count: int):
    """Panel query + timing + jumlah hasil."""
    header = Text()
    header.append("  QUERY  ", style="bold black on cyan")
    header.append(f"  {query}\n", style="bold white")
    header.append(f"  ⏱ Retrieval + Reranking: ", style="dim")
    header.append(f"{retrieval_ms:.0f}ms", style="bold yellow")
    header.append(f"  │  ", style="dim")
    header.append(f"{result_count} hasil ditemukan", style="bold green")

    console.print(Panel(
        header,
        border_style="cyan",
        box=box.HEAVY,
        padding=(0, 1),
    ))


def render_result_card(rank: int, result: RetrievalResult):
    """Satu hasil retrieval sebagai panel bergaya."""

    # ── Title: rank + rerank score ────────────────────────
    title = Text()
    title.append(f" [{rank}] ", style="bold cyan")
    title.append(f"rerank={result.score:.4f}", style="bold yellow")
    title.append(f"  embed={result.embed_score:.4f}", style="dim yellow")

    # ── Body ──────────────────────────────────────────────
    body = Text(overflow="fold")

    # Sumber (dari metadata JSONL)
    sumber = result.metadata.get("sumber", result.metadata.get("Sumber", "unknown"))
    body.append("  📄 ", style="dim")
    body.append(f"{sumber}\n", style="bold blue")

    # Metadata breadcrumbs: Kategori › Konteks_Lengkap (Bab | Pasal | Ayat)
    breadcrumbs = []

    kategori = result.metadata.get("kategori", result.metadata.get("Kategori", ""))
    if kategori:
        breadcrumbs.append(kategori)

    # Konteks_Lengkap — field baru dari JSONL
    konteks = result.metadata.get("konteks_lengkap", result.metadata.get("Konteks_Lengkap", ""))
    if konteks:
        breadcrumbs.append(konteks)

    # Fallback ke header lama jika Konteks_Lengkap kosong
    if not konteks:
        for hkey in ("header_1", "header_2", "header_3", "page_number"):
            hval = result.metadata.get(hkey.lower()) or result.metadata.get(hkey)
            if hval:
                breadcrumbs.append(str(hval))

    if breadcrumbs:
        body.append("  ", style="dim")
        for i, bc in enumerate(breadcrumbs):
            if i > 0:
                body.append(" › ", style="dim white")
            body.append(bc, style="dim italic")
        body.append("\n", style="dim")

    # Catatan
    catatan = result.metadata.get("catatan", result.metadata.get("Catatan / Anotasi", ""))
    if catatan:
        body.append(f"  📝 {catatan}\n", style="dim yellow")


    # ── Full text content ────────────────────────────────
    body.append("\n", style="dim")
    body.append(f"  {result.text.strip()}", style="green")

    console.print(Panel(
        body,
        title=title,
        title_align="left",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


# ============================================================
# INTERACTIVE CLI
# ============================================================

def main():
    """Loop interaktif: cari dokumen, tampilkan hasil retrieval."""

    silence_loggers()
    console.clear()
    render_header()

    # Load retriever (once)
    console.print()
    with console.status("[bold cyan]⏳ Memuat model embedding + reranker...", spinner="dots"):
        retriever = PolicyRetriever()
    console.print("[bold green]  ✅ Model siap![/]\n")

    while True:
        try:
            console.print(Rule(style="dim cyan"))
            query = console.input("[bold cyan]  🔎 Search › [/]").strip()

            if query.lower() in ("exit", "quit", "q", "keluar"):
                console.print()
                console.print(Panel(
                    "[bold white]👋 Terima kasih!\n"
                    "[dim]   Tim 4 Universitas Brawijaya[/]",
                    border_style="cyan",
                    box=box.DOUBLE_EDGE,
                    padding=(1, 2),
                ))
                break

            if not query:
                console.print("[dim yellow]   ⚠ Query kosong.[/]")
                continue

            # ── Retrieval ─────────────────────────────────
            start = time.time()
            results = retriever.retrieve(query)
            elapsed_ms = (time.time() - start) * 1000

            console.print()

            if not results:
                console.print(Panel(
                    "[yellow]Tidak ada dokumen relevan yang ditemukan.[/]",
                    title="[bold yellow]⚠ Tidak Ada Hasil",
                    border_style="yellow",
                    box=box.ROUNDED,
                ))
                continue

            # ── Display ───────────────────────────────────
            render_query_header(query, elapsed_ms, len(results))
            console.print()

            for i, r in enumerate(results, 1):
                render_result_card(i, r)

            console.print()

        except KeyboardInterrupt:
            console.print("\n[bold cyan]👋 Pencarian dihentikan.[/]")
            break
        except Exception as e:
            console.print(f"\n[bold red]   ❌ Error: {e}[/]\n")


if __name__ == "__main__":
    main()
