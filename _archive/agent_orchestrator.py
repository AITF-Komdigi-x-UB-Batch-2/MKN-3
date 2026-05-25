# ============================================================
# agent_orchestrator.py — Agentic RAG Orchestrator
# Social Welfare Policy Recommender System (Tim 3)
#
# Mengubah alur linear (retrieve → rerank → generate) menjadi
# agent loop yang bisa memutuskan:
#   - RETRIEVE: semantic search + reranking
#   - EXPAND:   naikkan top_k jika skor rendah
#   - GENERATE: panggil LLM untuk jawaban
#   - DONE:     selesai (termasuk fallback + disclaimer)
#
# Decision-making berbasis HEURISTIK SKOR (tanpa LLM tambahan).
# LLM (qwen-MKN1) hanya dipanggil saat GENERATE.
#
# Desain: CLI dulu, fungsi bisa di-import ke FastAPI.
# ============================================================

import time
import logging
import importlib
from dataclasses import dataclass, field
from enum import Enum

# IMPORT CONFIG FIRST to ensure HF_HOME (D: drive) is set before HuggingFace loads!
from config import (
    OLLAMA_BASE_URL, OLLAMA_SCORING_MODEL, OLLAMA_MODEL, OLLAMA_TEMPERATURE,
    SCORING_SYSTEM_PROMPT, SYSTEM_PROMPT,
    SCORING_PROMPT_TEMPLATE, PROMPT_TEMPLATE, POLICY_PROMPT_TEMPLATE,
    QDRANT_COLLECTION, EMBED_MODEL_NAME, RERANKER_MODEL_NAME,


    RETRIEVAL_TOP_K, RERANK_TOP_N,
    RELEVANCE_THRESHOLD, MAX_RETRIES,
    AGENT_MAX_LOOPS, EXPAND_TOP_K_STEP,
)


from langchain_ollama import OllamaLLM

# Import retriever (handle nama file dengan angka)
_mod_05 = importlib.import_module("05_retrieval_reranking")
PolicyRetriever = _mod_05.PolicyRetriever
RetrievalResult = _mod_05.RetrievalResult


# Reuse singkatan dari 03_normalize_jsonl.py
_mod_03 = importlib.import_module("03_normalize_jsonl")
ABBREVIATIONS = _mod_03.ABBREVIATIONS

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
# AGENT ACTIONS
# ============================================================

class Action(str, Enum):
    SCORE     = "SCORE"
    RETRIEVE  = "RETRIEVE"
    EXPAND    = "EXPAND"
    RECOMMEND = "RECOMMEND"
    DONE      = "DONE"


# ============================================================
# AGENT STATE
# ============================================================

@dataclass
class AgentState:
    """Seluruh state yang dipegang agent selama satu sesi query."""
    original_query: str
    working_query: str = ""
    scoring_result: str = ""
    candidates: list[RetrievalResult] = field(default_factory=list)
    finalists: list[RetrievalResult] = field(default_factory=list)
    policy_result: str = ""
    answer: str = ""
    disclaimer: str = ""

    # Tracking
    current_top_k: int = RETRIEVAL_TOP_K
    retrieval_attempts: int = 0
    loop_count: int = 0
    action_log: list[str] = field(default_factory=list)

    # Metrics
    top_score: float = 0.0
    avg_score: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0

    def log(self, action: str, detail: str = ""):
        entry = f"[Loop {self.loop_count}] {action}"
        if detail:
            entry += f": {detail}"
        self.action_log.append(entry)
        logger.info("🤖 %s", entry)


# ============================================================
# QUERY REFORMULATOR — Token-efficient (tanpa LLM)
# ============================================================

class QueryReformulator:
    """
    Reformulasi query tanpa memanggil LLM.
    - Ekspansi singkatan (reuse dari 03_normalize_jsonl.py)
    - Normalisasi whitespace
    """

    @staticmethod
    def expand_abbreviations(query: str) -> str:
        """Ekspansi singkatan yang umum di domain bansos."""
        result = query
        for abbr, full in ABBREVIATIONS.items():
            # Hanya expand jika singkatan berdiri sendiri
            pattern = r"\b" + re.escape(abbr) + r"\b"
            if re.search(pattern, result):
                result = re.sub(pattern, f"{full} ({abbr})", result, count=1)
        return result

    @staticmethod
    def reformulate(query: str, attempt: int) -> str:
        """
        Reformulasi query berdasarkan attempt number.
        Attempt 0: query asli (dengan ekspansi singkatan)
        Attempt 1: tambahkan konteks domain
        """
        expanded = QueryReformulator.expand_abbreviations(query)

        if attempt == 0:
            return expanded
        elif attempt == 1:
            # Tambahkan konteks domain jika belum ada
            domain_hint = "kebijakan sosial bantuan kemiskinan Indonesia"
            if "kebijakan" not in expanded.lower():
                return f"{expanded} {domain_hint}"
            return expanded
        else:
            return expanded


import re  # Dibutuhkan oleh QueryReformulator


# ============================================================
# QUALITY EVALUATOR — Heuristik skor
# ============================================================

class QualityEvaluator:
    """
    Evaluasi kualitas hasil retrieval berbasis skor numerik.
    Tanpa LLM call — murni heuristik.
    """

    @staticmethod
    def evaluate(
        finalists: list[RetrievalResult],
        threshold: float = RELEVANCE_THRESHOLD,
    ) -> dict:
        """
        Returns dict: {
            "quality": "good" | "low" | "empty",
            "top_score": float,
            "avg_score": float,
            "score_spread": float,
        }
        """
        if not finalists:
            return {
                "quality": "empty",
                "top_score": 0.0,
                "avg_score": 0.0,
                "score_spread": 0.0,
            }

        scores = [f.score for f in finalists]
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)
        score_spread = top_score - min(scores)

        if top_score >= threshold:
            quality = "good"
        else:
            quality = "low"

        return {
            "quality": quality,
            "top_score": top_score,
            "avg_score": avg_score,
            "score_spread": score_spread,
        }


# ============================================================
# CONTEXT BUILDER — Reuse dari 06_generation_v2.py
# ============================================================

def build_context(results: list[RetrievalResult]) -> str:
    """Bangun string konteks dari hasil retrieval untuk prompt LLM."""
    if not results:
        return "(Tidak ada dokumen relevan yang ditemukan.)"

    parts = []
    for i, r in enumerate(results, 1):
        sumber = r.metadata.get("sumber", r.metadata.get("Sumber", "unknown"))
        kategori = r.metadata.get("kategori", r.metadata.get("Kategori", "-"))
        
        # Ekstrak hirarki jika ada
        bab = r.metadata.get("bab", "")
        pasal = r.metadata.get("pasal", "")
        ayat = r.metadata.get("ayat", "")
        
        hierarchy = []
        if bab: hierarchy.append(bab)
        if pasal: hierarchy.append(pasal)
        if ayat: hierarchy.append(ayat)
        hierarchy_str = " | ".join(hierarchy) if hierarchy else "Konten Umum"

        header = f"[Dokumen {i}] (Sumber: {sumber} | {hierarchy_str} | Kategori: {kategori})"
        parts.append(f"{header}\n{r.text}")



    return "\n\n---\n\n".join(parts)


# ============================================================
# AGENT ORCHESTRATOR
# ============================================================

class AgentOrchestrator:
    """
    Agentic RAG — Decision loop berbasis heuristik.
    Satu LLM (qwen-MKN1) untuk generation saja.

    Alur:
        1. RETRIEVE: semantic search + rerank
        2. Evaluasi kualitas skor
        3. Jika rendah → EXPAND (naikkan top_k) → kembali ke 1
        4. Jika cukup → GENERATE (panggil LLM)
        5. DONE (atau fallback + disclaimer)

    Usage (importable):
        orch = AgentOrchestrator()
        result = orch.run("Kriteria penerima PKH")
        print(result.answer)
    """

    def __init__(self):
        logger.info("🤖 Inisialisasi Agent Orchestrator (Multi-Agent) ...")
        self.retriever = PolicyRetriever()
        
        # Agent 1: Scoring Agent (qwen-MKN1)
        self.scoring_llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_SCORING_MODEL,
            temperature=OLLAMA_TEMPERATURE,
            num_ctx=4096,
            repeat_penalty=1.15,
        )
        
        # Agent 2: Policy Agent (qwen2.5)
        self.policy_llm = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,

            temperature=OLLAMA_TEMPERATURE,
            num_ctx=8192,
            repeat_penalty=1.15,
        )
        
        self.reformulator = QueryReformulator()
        self.evaluator = QualityEvaluator()
        logger.info("✅ Agent Orchestrator siap.")

    # ──────────────────────────────────────────────────────
    # DECISION ENGINE
    # ──────────────────────────────────────────────────────

    def _decide(self, state: AgentState) -> Action:
        """
        Tentukan aksi berikutnya berdasarkan state saat ini.
        Murni heuristik — tanpa LLM call.
        """
        # Safety brake
        if state.loop_count >= AGENT_MAX_LOOPS:
            state.log("SAFETY_BRAKE", f"Max loops ({AGENT_MAX_LOOPS}) tercapai")
            return Action.RECOMMEND  # force recommend

        # Step 1: Belum ada scoring
        if not state.scoring_result:
            return Action.SCORE

        # Step 2: Belum pernah retrieve
        if state.retrieval_attempts == 0:
            return Action.RETRIEVE

        # Evaluasi kualitas hasil
        eval_result = self.evaluator.evaluate(state.finalists)
        state.top_score = eval_result["top_score"]
        state.avg_score = eval_result["avg_score"]

        if eval_result["quality"] == "empty":
            # Tidak ada hasil sama sekali
            if state.retrieval_attempts < MAX_RETRIES:
                return Action.EXPAND
            return Action.DONE  # fallback

        if eval_result["quality"] == "low":
            # Skor rendah — coba expand
            if state.retrieval_attempts < MAX_RETRIES:
                state.log(
                    "EVAL_LOW",
                    f"top={eval_result['top_score']:.4f} < threshold={RELEVANCE_THRESHOLD}",
                )
                return Action.EXPAND
            # Sudah retry max — recommend anyway (best effort)
            state.disclaimer = (
                "⚠️ Catatan: Konteks yang ditemukan memiliki relevansi rendah "
                f"(skor tertinggi: {eval_result['top_score']:.4f}). "
                "Rekomendasi ini mungkin kurang akurat."
            )
            return Action.RECOMMEND

        # Kualitas baik → recommend
        return Action.RECOMMEND

    # ──────────────────────────────────────────────────────
    # ACTION EXECUTORS
    # ──────────────────────────────────────────────────────

    def _do_retrieve(self, state: AgentState):
        """Execute RETRIEVE: semantic search + rerank."""
        state.retrieval_attempts += 1

        # Reformulasi query berdasarkan attempt
        state.working_query = self.reformulator.reformulate(
            state.original_query,
            state.retrieval_attempts - 1,
        )

        state.log(
            Action.RETRIEVE,
            f"query='{state.working_query[:80]}...' top_k={state.current_top_k}",
        )

        t0 = time.time()
        state.finalists = self.retriever.retrieve(
            state.working_query,
            top_k=state.current_top_k,
            top_n=RERANK_TOP_N,
        )
        state.retrieval_time += time.time() - t0

        state.log(
            "RESULT",
            f"{len(state.finalists)} finalis, "
            f"top_score={state.finalists[0].score:.4f}" if state.finalists else "0 finalis",
        )

    def _do_expand(self, state: AgentState):
        """Execute EXPAND: naikkan top_k dan re-retrieve."""
        old_k = state.current_top_k
        state.current_top_k += EXPAND_TOP_K_STEP
        state.log(Action.EXPAND, f"top_k: {old_k} → {state.current_top_k}")
        self._do_retrieve(state)

    def _do_score(self, state: AgentState) -> str:
        """Execute SCORE: panggil Scoring Agent (MKN1)."""
        state.log(Action.SCORE, f"Memanggil Scoring Agent ({OLLAMA_SCORING_MODEL})")
        
        prompt = SCORING_PROMPT_TEMPLATE.format(
            system_prompt=SCORING_SYSTEM_PROMPT,
            query=state.original_query,
        )

        t0 = time.time()
        answer = self.scoring_llm.invoke(prompt)
        state.generation_time += time.time() - t0

        if isinstance(answer, str):
            state.scoring_result = answer
        else:
            state.scoring_result = answer.content

        return state.scoring_result

    def _do_recommend(self, state: AgentState) -> str:
        """Execute RECOMMEND: panggil Policy Agent (Qwen2.5) dengan konteks."""
        state.log(Action.RECOMMEND, f"Memanggil Policy Agent ({OLLAMA_MODEL}) dengan {len(state.finalists)} finalis")

        context = build_context(state.finalists)
        prompt = POLICY_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            scoring_result=state.scoring_result,
            context=context,
        )



        t0 = time.time()
        answer = self.policy_llm.invoke(prompt)
        state.generation_time += time.time() - t0

        if isinstance(answer, str):
            state.policy_result = answer
        else:
            state.policy_result = answer.content

        # Combine answers
        combined = f"{state.scoring_result}\n\n{state.policy_result}"
        
        # Tambahkan disclaimer jika ada
        if state.disclaimer:
            state.answer = f"{state.disclaimer}\n\n{combined}"
        else:
            state.answer = combined

        return state.answer

    # ──────────────────────────────────────────────────────
    # MAIN RUN LOOP
    # ──────────────────────────────────────────────────────

    def run(self, query: str) -> AgentState:
        """
        Jalankan agent loop untuk satu query.
        Returns AgentState lengkap (untuk API / CLI).
        """
        state = AgentState(original_query=query, working_query=query)
        t_start = time.time()

        while state.loop_count < AGENT_MAX_LOOPS:
            state.loop_count += 1
            action = self._decide(state)

            if action == Action.RETRIEVE:
                self._do_retrieve(state)

            elif action == Action.EXPAND:
                self._do_expand(state)

            elif action == Action.SCORE:
                self._do_score(state)

            elif action == Action.RECOMMEND:
                self._do_recommend(state)
                state.log(Action.DONE, f"Jawaban dihasilkan ({len(state.answer)} chars)")
                break

            elif action == Action.DONE:
                # Fallback: tidak ada konteks
                state.answer = (
                    "⚠️ Maaf, sistem tidak menemukan dokumen yang cukup relevan "
                    "untuk menjawab pertanyaan Anda.\n\n"
                    "Saran:\n"
                    "• Coba gunakan kata kunci yang lebih spesifik\n"
                    "• Sebutkan nama program, regulasi, atau pasal tertentu\n"
                    "• Contoh: 'Kriteria penerima PKH menurut Permensos 1/2018'"
                )
                state.disclaimer = "Tidak ada konteks relevan ditemukan."
                state.log(Action.DONE, "Fallback — tidak ada konteks relevan")
                break

        state.total_time = time.time() - t_start
        return state

    def run_stream(self, query: str, on_token=None) -> AgentState:
        """
        Versi streaming — panggil on_token(token) untuk setiap token LLM.
        Cocok untuk CLI (Rich) atau SSE di FastAPI.
        """
        state = AgentState(original_query=query, working_query=query)
        t_start = time.time()

        while state.loop_count < AGENT_MAX_LOOPS:
            state.loop_count += 1
            action = self._decide(state)

            if action == Action.RETRIEVE:
                self._do_retrieve(state)

            elif action == Action.EXPAND:
                self._do_expand(state)

            elif action == Action.SCORE:
                state.log(Action.SCORE, f"Streaming dari Scoring Agent ({OLLAMA_SCORING_MODEL})")
                prompt = SCORING_PROMPT_TEMPLATE.format(
                    system_prompt=SCORING_SYSTEM_PROMPT,
                    query=state.original_query,
                )

                t0 = time.time()
                tokens = []
                for chunk in self.scoring_llm.stream(prompt):
                    token = chunk if isinstance(chunk, str) else chunk.content
                    tokens.append(token)
                    if on_token:
                        on_token(token)

                state.generation_time += time.time() - t0
                state.scoring_result = "".join(tokens)
                
                # Tambahkan spasi/newline antara output scoring dan policy agar UI terlihat rapi
                if on_token:
                    on_token("\n\n")

            elif action == Action.RECOMMEND:
                state.log(Action.RECOMMEND, f"Streaming dari Policy Agent ({OLLAMA_MODEL}) dengan {len(state.finalists)} finalis")
                context = build_context(state.finalists)
                prompt = POLICY_PROMPT_TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    scoring_result=state.scoring_result,
                    context=context,
                )



                # Stream disclaimer dulu jika ada
                if state.disclaimer and on_token:
                    on_token(state.disclaimer + "\n\n")

                t0 = time.time()
                tokens = []
                for chunk in self.policy_llm.stream(prompt):
                    token = chunk if isinstance(chunk, str) else chunk.content
                    tokens.append(token)
                    if on_token:
                        on_token(token)

                state.generation_time += time.time() - t0
                state.policy_result = "".join(tokens)
                
                combined = f"{state.scoring_result}\n\n{state.policy_result}"
                if state.disclaimer:
                    state.answer = f"{state.disclaimer}\n\n{combined}"
                else:
                    state.answer = combined
                    
                state.log(Action.DONE, f"Stream selesai ({len(state.answer)} chars)")
                break

            elif action == Action.DONE:
                fallback = (
                    "⚠️ Maaf, sistem tidak menemukan dokumen yang cukup relevan "
                    "untuk menjawab pertanyaan Anda.\n\n"
                    "Saran:\n"
                    "• Coba gunakan kata kunci yang lebih spesifik\n"
                    "• Sebutkan nama program, regulasi, atau pasal tertentu"
                )
                state.answer = fallback
                state.disclaimer = "Tidak ada konteks relevan ditemukan."
                if on_token:
                    on_token(fallback)
                state.log(Action.DONE, "Fallback — tidak ada konteks relevan")
                break

        state.total_time = time.time() - t_start
        return state


# ============================================================
# CLI — Interactive (Rich UI)
# ============================================================

def interactive_cli():
    """CLI interaktif dengan Rich UI — mirip 06_generation_v2.py."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich import box

    console = Console()

    # Silence semua logger agar hanya Rich UI
    logging.getLogger().setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)

    console.clear()

    # ── Header ────────────────────────────────────────────
    header = Text()
    header.append("  AGENTIC RAG", style="bold cyan")
    header.append("  ░▒▓  ", style="dim cyan")
    header.append("Kebijakan Sosial", style="bold white")
    header.append("  ░▒▓  ", style="dim cyan")
    header.append("Tim 3 UB\n", style="bold white")

    info = Text()
    info.append("  ◈ Mode      ", style="dim")
    info.append("Agentic (heuristik decision loop)\n", style="bold yellow")
    info.append("  ◈ Scoring   ", style="dim")
    info.append(f"{OLLAMA_SCORING_MODEL}\n", style="green")
    info.append("  ◈ Policy    ", style="dim")
    info.append(f"{OLLAMA_MODEL}\n", style="green")

    info.append("  ◈ Embedding ", style="dim")
    info.append(f"{EMBED_MODEL_NAME}\n", style="green")
    info.append("  ◈ Reranker  ", style="dim")
    info.append(f"{RERANKER_MODEL_NAME}\n", style="green")
    info.append("  ◈ Collection ", style="dim")
    info.append(f"{QDRANT_COLLECTION}\n", style="green")
    info.append("  ◈ Threshold ", style="dim")
    info.append(f"{RELEVANCE_THRESHOLD}", style="yellow")
    info.append(f"  │  Max retries={MAX_RETRIES}  Max loops={AGENT_MAX_LOOPS}\n", style="dim")
    info.append("\n  Ketik pertanyaan. Ketik ", style="dim white")
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

    # ── Inisialisasi Agent ────────────────────────────────
    console.print()
    with console.status("[bold cyan]⏳ Memuat model...", spinner="dots"):
        agent = AgentOrchestrator()
    console.print("[bold green]  ✅ Agent siap![/]\n")

    # ── Loop Interaktif ───────────────────────────────────
    while True:
        try:
            console.print(Rule(style="dim cyan"))
            query = console.input("[bold cyan]  🔎 Pertanyaan › [/]").strip()

            if query.lower() in ("exit", "quit", "q", "keluar"):
                console.print()
                console.print(Panel(
                    "[bold white]👋 Terima kasih!\n"
                    "[dim]   Agentic RAG — Tim 3 Universitas Brawijaya[/]",
                    border_style="cyan",
                    box=box.DOUBLE_EDGE,
                    padding=(1, 2),
                ))
                break

            if not query:
                console.print("[dim yellow]   ⚠ Pertanyaan kosong.[/]")
                continue

            # ── Run Agent (streaming) ─────────────────────
            console.print()

            # Tampilkan status agent decisions
            phase_text = Text()
            phase_text.append("  🤖 AGENT LOOP\n", style="bold yellow")

            def on_token(token):
                console.print(f"[green]{token}[/]", end="", highlight=False)

            # Run agent
            state = agent.run_stream(query, on_token=on_token)

            console.print()  # newline setelah streaming

            # ── Display: Agent Decision Log ───────────────
            if state.action_log:
                log_text = Text()
                for entry in state.action_log:
                    log_text.append(f"  {entry}\n", style="dim")
                console.print(Panel(
                    log_text,
                    title="[dim]Agent Decision Log[/]",
                    title_align="left",
                    border_style="dim yellow",
                    box=box.ROUNDED,
                    padding=(0, 1),
                ))

            # ── Display: Retrieval Results ────────────────
            if state.finalists:
                for i, r in enumerate(state.finalists[:3], 1):
                    sumber = r.metadata.get("sumber", r.metadata.get("Sumber", "unknown"))
                    kategori = r.metadata.get("kategori", r.metadata.get("Kategori", "-"))
                    
                    title_t = Text()
                    title_t.append(f" [{i}] ", style="bold cyan")
                    title_t.append(f"rerank={r.score:.4f}", style="bold yellow")
                    title_t.append(f"  embed={r.embed_score:.4f}", style="dim yellow")

                    body = Text()
                    body.append(f"  📄 {sumber}\n", style="bold blue")
                    body.append(f"  Kategori: {kategori}\n", style="dim italic")
                    
                    preview = r.text[:250].strip()
                    if len(r.text) > 250:
                        preview += " ..."
                    body.append(f"  {preview}", style="green")


                    console.print(Panel(
                        body,
                        title=title_t,
                        title_align="left",
                        border_style="dim cyan",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    ))

            # ── Footer ───────────────────────────────────
            console.print(Rule(style="green"))
            footer = Text()
            footer.append(f"  ⏱ Total: {state.total_time:.1f}s", style="dim")
            footer.append(f"  │  Retrieval: {state.retrieval_time:.1f}s", style="dim")
            footer.append(f"  │  Generation: {state.generation_time:.1f}s", style="dim")
            footer.append(f"  │  Loops: {state.loop_count}", style="dim")
            footer.append(f"  │  Retries: {state.retrieval_attempts}", style="dim")
            console.print(footer)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[bold cyan]👋 Agent dihentikan.[/]")
            break
        except Exception as e:
            console.print(f"\n[bold red]   ❌ Error: {e}[/]")
            console.print("[dim]   Silakan coba pertanyaan lain.[/]\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    interactive_cli()
