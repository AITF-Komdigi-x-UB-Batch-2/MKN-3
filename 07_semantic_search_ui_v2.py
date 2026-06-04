# ============================================================
# 07_semantic_search_ui_v2.py — Semantic Search UI v2 (CSV payload)
# Social Welfare Policy Recommender System (Tim 4)
#
# Menampilkan hasil retrieval + reranking tanpa LLM generation.
# v2: Metadata Bab/Pasal/Ayat/Konteks_Lengkap dari kolom CSV
# ============================================================

import os
import time
import logging
import textwrap

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.table import Table
from rich.columns import Columns
from rich.padding import Padding
from rich.align import Align
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

console = Console(highlight=False)

THEME = {
    "primary"    : "bold #00D4FF",
    "secondary"  : "#0099BB",
    "accent"     : "bold #FFB300",
    "accent_dim" : "#997000",
    "success"    : "#00E676",
    "source"     : "bold #64B5F6",
    "breadcrumb" : "#78909C",
    "content"    : "#E0F7E9",
    "dim"        : "#546E7A",
    "warn"       : "#FFA726",
    "error"      : "bold #FF5252",
    "border_main": "cyan",
    "border_card": "#1A6B7A",
    "rank_hi"    : "bold #FFD700",
    "rank_lo"    : "#B8860B",
    "score_bar"  : "#00BCD4",
}


def silence_loggers():
    logging.getLogger().setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)


def _wrap(text: str, indent: int = 2) -> str:
    """Wrap teks agar pas di lebar terminal, hindari clipping."""
    width = max(40, console.width - 10 - indent)
    pad   = " " * indent
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            wrapped = textwrap.fill(paragraph, width=width,
                                    subsequent_indent=pad,
                                    break_long_words=False,
                                    break_on_hyphens=True)
            lines.append(pad + wrapped.lstrip())
    return "\n".join(lines)


def _score_bar(score: float, width: int = 12) -> str:
    """Visualisasi skor sebagai progress bar mini."""
    filled = round(score * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


# ============================================================
# DISPLAY COMPONENTS
# ============================================================

def render_header():
    """Header utama aplikasi."""
    # ── Banner ────────────────────────────────────────────
    banner = Text(justify="center")
    banner.append("\n")
    banner.append("╔══════════════════════════════════════╗\n", style="cyan")
    banner.append("║  ", style="cyan")
    banner.append("⚡ SEMANTIC SEARCH", style="bold #00D4FF")
    banner.append("  ·  ", style="dim cyan")
    banner.append("Kebijakan Sosial", style="bold white")
    banner.append("  ║\n", style="cyan")
    banner.append("║  ", style="cyan")
    banner.append("Tim 4  ·  Universitas Brawijaya        ", style="dim white")
    banner.append("║\n", style="cyan")
    banner.append("╚══════════════════════════════════════╝\n", style="cyan")

    # ── Config table ──────────────────────────────────────
    cfg = Table(box=None, padding=(0, 2, 0, 0), show_header=False,
                min_width=44)
    cfg.add_column("key",   style=THEME["dim"],      no_wrap=True)
    cfg.add_column("value", style=THEME["success"],  no_wrap=False)

    cfg.add_row("Embedding  ▸", EMBED_MODEL_NAME)
    cfg.add_row("Reranker   ▸", RERANKER_MODEL_NAME)
    cfg.add_row("Collection ▸", QDRANT_COLLECTION)
    cfg.add_row("Top-K / Top-N ▸",
                f"{RETRIEVAL_TOP_K} retrieved  →  {RERANK_TOP_N} reranked")

    # ── Hint ──────────────────────────────────────────────
    hint = Text()
    hint.append("\n  Ketik query lalu tekan ", style=THEME["dim"])
    hint.append("Enter", style="bold white")
    hint.append("  ·  Ketik ", style=THEME["dim"])
    hint.append("exit", style=THEME["error"])
    hint.append(" untuk keluar\n", style=THEME["dim"])

    console.print(Panel(
        Align.center(banner),
        box=box.DOUBLE_EDGE,
        border_style=THEME["border_main"],
        padding=(0, 2),
    ))
    console.print(Panel(
        cfg,
        box=box.SIMPLE,
        border_style=THEME["secondary"],
        padding=(0, 3),
    ))
    console.print(hint)


def render_query_bar(query: str, retrieval_ms: float, result_count: int):
    """Bar ringkasan query + statistik."""
    t = Text()
    t.append("  QUERY  ", style="bold black on #00D4FF")
    t.append("  ", style="")
    t.append(query, style="bold white")
    t.append("\n\n", style="")

    # stats row
    t.append("  ⏱ ", style=THEME["dim"])
    t.append(f"{retrieval_ms:.0f} ms", style=THEME["accent"])
    t.append("   ·   ", style=THEME["dim"])
    t.append(f"{result_count} chunk", style=THEME["success"])
    t.append(" ditemukan", style=THEME["dim"])

    console.print()
    console.print(Panel(
        t,
        box=box.HEAVY,
        border_style=THEME["border_main"],
        padding=(0, 1),
    ))
    console.print()


def render_result_card(rank: int, result: RetrievalResult, total: int):
    """Kartu hasil retrieval — teks penuh, tidak terpotong."""
    w = console.width

    # ── Scores ────────────────────────────────────────────
    rerank_bar = _score_bar(result.score)
    embed_bar  = _score_bar(result.embed_score)

    score_tbl = Table(box=None, padding=(0, 1, 0, 0), show_header=False)
    score_tbl.add_column("lbl",  style=THEME["dim"],       no_wrap=True)
    score_tbl.add_column("bar",  style=THEME["score_bar"], no_wrap=True)
    score_tbl.add_column("val",  style=THEME["accent"],    no_wrap=True)
    score_tbl.add_row("rerank", rerank_bar, f"{result.score:.4f}")
    score_tbl.add_row("embed ",  embed_bar,  f"{result.embed_score:.4f}")

    # ── Source & breadcrumbs ───────────────────────────────
    sumber = result.metadata.get("sumber",
             result.metadata.get("Sumber", "unknown"))

    src_text = Text()
    src_text.append("  📄 ", style=THEME["dim"])
    src_text.append(sumber, style=THEME["source"])

    breadcrumbs = []
    kategori = result.metadata.get("kategori",
               result.metadata.get("Kategori", ""))
    if kategori:
        breadcrumbs.append(kategori)

    konteks = result.metadata.get("konteks_lengkap",
              result.metadata.get("Konteks_Lengkap", ""))
    if konteks:
        breadcrumbs.append(konteks)

    if not konteks:
        for hkey in ("header_1", "header_2", "header_3", "page_number"):
            hval = (result.metadata.get(hkey.lower())
                    or result.metadata.get(hkey))
            if hval:
                breadcrumbs.append(str(hval))

    bc_text = Text()
    if breadcrumbs:
        bc_text.append("  ╰─ ", style=THEME["dim"])
        for i, bc in enumerate(breadcrumbs):
            if i > 0:
                bc_text.append("  ›  ", style=THEME["dim"])
            bc_text.append(bc, style=THEME["breadcrumb"])

    catatan = result.metadata.get("catatan",
              result.metadata.get("Catatan / Anotasi", ""))

    # ── Full content (wrapped, no clipping) ───────────────
    wrapped = _wrap(result.text.strip(), indent=2)

    content_text = Text()
    content_text.append(wrapped, style=THEME["content"])

    # ── Assemble body ─────────────────────────────────────
    body = Text()

    # Score section
    body.append_text(Text.from_markup(
        f"  [dim]rerank[/] [{THEME['score_bar']}]{rerank_bar}[/]"
        f"  [{THEME['accent']}]{result.score:.4f}[/]"
        f"    [dim]embed[/] [{THEME['score_bar']}]{embed_bar}[/]"
        f"  [{THEME['accent_dim']}]{result.embed_score:.4f}[/]\n"
    ))

    body.append("\n")

    # Source
    body.append("  📄 ", style=THEME["dim"])
    body.append(sumber + "\n", style=THEME["source"])

    # Breadcrumbs
    if breadcrumbs:
        body.append("  ╰─ ", style=THEME["dim"])
        for i, bc in enumerate(breadcrumbs):
            if i > 0:
                body.append("  ›  ", style=THEME["dim"])
            body.append(bc, style=THEME["breadcrumb"])
        body.append("\n")

    # Catatan
    if catatan:
        cat_wrapped = _wrap(catatan, indent=5)
        body.append(f"\n  📝 ", style=THEME["warn"])
        body.append(cat_wrapped.lstrip() + "\n", style=THEME["warn"])

    # Divider
    body.append("\n")
    body.append("  " + "─" * (w - 14) + "\n", style=THEME["dim"])
    body.append("\n")

    # Content — fully wrapped
    body.append(wrapped + "\n", style=THEME["content"])

    # ── Rank label ────────────────────────────────────────
    rank_style = THEME["rank_hi"] if rank <= 3 else THEME["rank_lo"]
    title = Text()
    title.append(f" #{rank} / {total} ", style=rank_style)

    console.print(Panel(
        body,
        title=title,
        title_align="left",
        box=box.ROUNDED,
        border_style=THEME["border_card"],
        padding=(0, 1),
    ))
    console.print()


def render_no_result():
    console.print(Panel(
        Text("  Tidak ada dokumen relevan ditemukan.", style=THEME["warn"]),
        title=Text(" ⚠ Kosong ", style=f"bold {THEME['warn']}"),
        border_style=THEME["warn"],
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def render_farewell():
    msg = Text(justify="center")
    msg.append("\n  👋  Terima kasih!\n", style="bold white")
    msg.append("  Tim 4  ·  Universitas Brawijaya\n", style=THEME["dim"])
    console.print()
    console.print(Panel(msg, border_style="cyan",
                        box=box.DOUBLE_EDGE, padding=(0, 2)))


# ============================================================
# INTERACTIVE CLI
# ============================================================

def main():
    silence_loggers()
    console.clear()
    render_header()

    with console.status(
        f"  [{THEME['primary']}]Memuat model embedding + reranker…[/]",
        spinner="dots12",
        spinner_style="cyan",
    ):
        retriever = PolicyRetriever()

    console.print(f"  [{THEME['success']}]✔ Model siap.[/]\n")

    while True:
        try:
            console.print(Rule(style=THEME["secondary"]))
            query = console.input(
                f"  [{THEME['primary']}]🔎  Search ›[/]  "
            ).strip()

            if query.lower() in ("exit", "quit", "q", "keluar"):
                render_farewell()
                break

            if not query:
                console.print(
                    f"  [{THEME['warn']}]⚠  Query kosong, coba lagi.[/]\n"
                )
                continue

            # ── Retrieval ─────────────────────────────────
            with console.status(
                f"  [{THEME['primary']}]Mencari…[/]",
                spinner="dots",
                spinner_style="cyan",
            ):
                start   = time.time()
                results = retriever.retrieve(query)
                elapsed = (time.time() - start) * 1000

            if not results:
                render_no_result()
                continue

            render_query_bar(query, elapsed, len(results))

            for i, r in enumerate(results, 1):
                render_result_card(i, r, len(results))

        except KeyboardInterrupt:
            console.print(
                f"\n  [{THEME['primary']}]👋  Pencarian dihentikan.[/]\n"
            )
            break
        except Exception as e:
            console.print(f"\n  [{THEME['error']}]❌  Error: {e}[/]\n")


if __name__ == "__main__":
    main()