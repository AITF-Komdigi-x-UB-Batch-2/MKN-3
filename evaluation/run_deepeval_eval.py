"""
Run LLM-judge RAG evaluation with DeepEval.

This complements evaluation/run_rag_eval.py:
- run_rag_eval.py gives deterministic string/regex checks.
- run_deepeval_eval.py gives LLM-judge metrics such as faithfulness and
  answer relevancy.

Examples:
    python evaluation/run_deepeval_eval.py --dry-run
    python evaluation/run_deepeval_eval.py --limit 3
    python evaluation/run_deepeval_eval.py --case-id GT_ASPD_001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import config di top-level agar env/cache aktif sebelum library AI lain
from config import (  # noqa: E402
    RUNPOD_API_KEY,
    RUNPOD_MODEL_NAME,
    RUNPOD_TEMPERATURE,
    TIM1_GENERATION_API_URL,
    TIM1_API_TIMEOUT_S,
)


GROUND_TRUTH_DEFAULT = Path(__file__).resolve().parent / "ground_truth_rag_cases.jsonl"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parent / "results" / "deepeval"


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_cases(
    cases: list[dict[str, Any]],
    case_id: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = cases
    if case_id:
        wanted = {part.strip() for part in case_id.split(",") if part.strip()}
        selected = [case for case in selected if case.get("case_id") in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_scoring_result(case: dict[str, Any]) -> str:
    return json.dumps(case.get("mock_tim1_output", {}), ensure_ascii=False, indent=2)


def build_case_input(case: dict[str, Any]) -> str:
    return (
        f"Case ID: {case.get('case_id', '-')}\n"
        f"Fokus: {case.get('focus', '-')}\n\n"
        f"Profil warga:\n{case.get('profil_warga', '')}\n\n"
        "Tugas: rekomendasikan status kelayakan program bantuan sosial utama, "
        "alasan, dan bukti sumber."
    )


def build_expected_output(case: dict[str, Any]) -> str:
    expected = case.get("expected_main_programs", {})
    lines = [
        f"Fokus kasus: {case.get('focus', '-')}",
        "",
        "Status program utama yang diharapkan:",
        f"- Eligible: {', '.join(expected.get('eligible', [])) or '-'}",
        f"- Mungkin eligible: {', '.join(expected.get('mungkin_eligible', [])) or '-'}",
        f"- Tidak eligible: {', '.join(expected.get('tidak_eligible', [])) or '-'}",
        f"- Ranking: {', '.join(expected.get('ranking', [])) or '-'}",
        "",
        "Reasoning wajib menyebut:",
    ]

    for item in case.get("expected_reasoning_must_mention", []):
        lines.append(f"- {item}")

    lines.extend(["", "Evidence wajib:"])
    for evidence in case.get("expected_evidence_requirements", []):
        lines.append(
            "- "
            f"{evidence.get('program', '-')} | "
            f"sumber memuat '{evidence.get('source_contains', '-')}' | "
            f"halaman {evidence.get('page_number', '-')} | "
            f"tipe {evidence.get('tipe', '-')}"
        )

    lines.extend(
        [
            "",
            "Batasan output:",
            "- Jangan merekomendasikan program tambahan di luar 6 program utama.",
            "- Jangan membuat bagian rekomendasi bantuan tambahan dari regulasi.",
            "- Jangan membuat bagian rekomendasi tindak lanjut atau catatan petugas.",
        ]
    )

    quality_checks = case.get("quality_checks", [])
    if quality_checks:
        lines.extend(["", "Quality checks:"])
        for item in quality_checks:
            lines.append(f"- {item}")

    return "\n".join(lines)


def format_retrieval_result(result: Any, max_chars: int) -> str:
    metadata = getattr(result, "metadata", {}) or {}
    text = getattr(result, "text", "")
    text = str(text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."

    parts = [
        f"Sumber: {metadata.get('sumber', '-')}",
        f"Halaman: {metadata.get('page_number', '-')}",
    ]
    if metadata.get("nama_bansos"):
        parts.append(f"Program: {metadata.get('nama_bansos')}")
    if metadata.get("tipe_konten"):
        parts.append(f"Tipe: {metadata.get('tipe_konten')}")
    if metadata.get("retrieval_priority"):
        parts.append(f"Priority: {metadata.get('retrieval_priority')}")

    return " | ".join(parts) + "\n" + text


class RetrievalRecorder:
    """
    Delegate wrapper that records every RetrievalResult used by generation.py.

    This keeps DeepEval's retrieval_context close to the actual context seen by
    RAGGenerator.recommend(), instead of running a separate approximate retrieval.
    """

    def __init__(self, retriever: Any):
        self._retriever = retriever
        self.results: list[Any] = []
        self._seen: set[str] = set()

    def reset(self) -> None:
        self.results.clear()
        self._seen.clear()

    def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
        results = self._retriever.retrieve(*args, **kwargs)
        for result in results:
            metadata = getattr(result, "metadata", {}) or {}
            key = "|".join(
                [
                    str(metadata.get("sumber", "")),
                    str(metadata.get("page_number", "")),
                    str(getattr(result, "text", ""))[:160],
                ]
            )
            if key not in self._seen:
                self._seen.add(key)
                self.results.append(result)
        return results

    def context_texts(self, max_contexts: int, max_context_chars: int) -> list[str]:
        return [
            format_retrieval_result(result, max_context_chars)
            for result in self.results[:max_contexts]
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._retriever, name)


@dataclass
class MetricResult:
    score: float | None
    reason: str
    success: bool | None


def load_deepeval_components():
    try:
        from deepeval.metrics import (  # type: ignore
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
        )
        from deepeval.models.base_model import DeepEvalBaseLLM  # type: ignore
        from deepeval.test_case import LLMTestCase  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DeepEval belum tersedia. Pastikan dependencies sudah diinstall: "
            "pip install -r requirements.txt"
        ) from exc

    return (
        DeepEvalBaseLLM,
        LLMTestCase,
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
    )


def build_api_judge(
    DeepEvalBaseLLM: type,
    model_name: str,
    api_url: str,
    api_key: str,
    json_mode: bool,
    temperature: float,
):
    class ApiDeepEval(DeepEvalBaseLLM):
        def __init__(
            self,
            model_name: str,
            api_url: str,
            api_key: str,
            json_mode: bool,
            temperature: float,
        ):
            self.model_name = model_name
            self.api_url = api_url
            self.api_key = api_key
            self.json_mode = json_mode
            self.temperature = temperature

        def load_model(self) -> str:
            return self.model_name

        def _messages(self, prompt: str) -> list[dict[str, str]]:
            if not self.json_mode:
                return [{"role": "user", "content": prompt}]
            return [
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation model. Return only valid JSON. "
                        "Do not use markdown fences, comments, or explanatory prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

        def _payload(self, prompt: str) -> dict[str, Any]:
            return {
                "model": self.model_name,
                "messages": self._messages(prompt),
                "temperature": self.temperature,
            }

        def _headers(self) -> dict[str, str]:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            return headers

        @staticmethod
        def _content(response_json: dict[str, Any]) -> str:
            return response_json["choices"][0]["message"]["content"]

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            import httpx

            with httpx.Client(timeout=TIM1_API_TIMEOUT_S) as client:
                response = client.post(
                    self.api_url,
                    json=self._payload(prompt),
                    headers=self._headers(),
                )
                response.raise_for_status()
                return self._content(response.json())

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            import httpx

            async with httpx.AsyncClient(timeout=TIM1_API_TIMEOUT_S) as client:
                response = await client.post(
                    self.api_url,
                    json=self._payload(prompt),
                    headers=self._headers(),
                )
                response.raise_for_status()
                return self._content(response.json())

        def get_model_name(self) -> str:
            return self.model_name

    return ApiDeepEval(
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        json_mode=json_mode,
        temperature=temperature,
    )


def build_metrics(args: argparse.Namespace, judge_model: Any, metric_classes: tuple[type, type, type]) -> list[Any]:
    FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric = metric_classes
    selected = {part.strip() for part in args.metrics.split(",") if part.strip()}

    metric_builders = {
        "faithfulness": lambda: FaithfulnessMetric(
            threshold=args.threshold,
            model=judge_model,
            include_reason=True,
            async_mode=args.async_metrics,
        ),
        "answer_relevancy": lambda: AnswerRelevancyMetric(
            threshold=args.threshold,
            model=judge_model,
            include_reason=True,
            async_mode=args.async_metrics,
        ),
        "contextual_precision": lambda: ContextualPrecisionMetric(
            threshold=args.threshold,
            model=judge_model,
            include_reason=True,
            async_mode=args.async_metrics,
        ),
    }

    unknown = selected - set(metric_builders)
    if unknown:
        raise SystemExit(f"Metric tidak dikenal: {', '.join(sorted(unknown))}")

    return [metric_builders[name]() for name in metric_builders if name in selected]


def measure_metrics(metrics: list[Any], test_case: Any) -> dict[str, MetricResult]:
    results: dict[str, MetricResult] = {}
    for metric in metrics:
        name = getattr(metric, "name", metric.__class__.__name__)
        normalized_name = name.lower().replace(" ", "_")
        try:
            metric.measure(test_case)
        except Exception as exc:
            results[normalized_name] = MetricResult(
                score=None,
                reason=f"ERROR: {exc}",
                success=False,
            )
            print(f"[WARN] Metric {normalized_name} gagal: {exc}")
            continue

        results[normalized_name] = MetricResult(
            score=getattr(metric, "score", None),
            reason=str(getattr(metric, "reason", "")),
            success=getattr(metric, "success", None),
        )
    return results


def run_cases(args: argparse.Namespace, cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    (
        DeepEvalBaseLLM,
        LLMTestCase,
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
    ) = load_deepeval_components()

    from generation import RAGGenerator

    judge_name = args.judge_model or RUNPOD_MODEL_NAME
    judge_model = build_api_judge(
        DeepEvalBaseLLM,
        judge_name,
        args.judge_api_url or TIM1_GENERATION_API_URL,
        args.judge_api_key or RUNPOD_API_KEY,
        json_mode=not args.no_judge_json_mode,
        temperature=args.judge_temperature,
    )
    metrics = build_metrics(
        args,
        judge_model,
        (FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric),
    )

    print(f"Judge model: {judge_name}")
    print(f"JSON mode  : {not args.no_judge_json_mode}")
    print(f"Async mode : {args.async_metrics}")
    print(f"Metrics    : {', '.join(getattr(m, 'name', m.__class__.__name__) for m in metrics)}")

    rag = RAGGenerator()
    recorder = RetrievalRecorder(rag.retriever)
    rag.retriever = recorder

    rows: list[dict[str, Any]] = []
    summary_scores: dict[str, list[float]] = {}

    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] {case['case_id']} - {case.get('focus', '')}")
        recorder.reset()
        start = time.time()

        scoring_result = "" if args.no_scoring_result else build_scoring_result(case)
        answer = rag.recommend(
            case["profil_warga"],
            scoring_result=scoring_result,
            stream=False,
            show_chunks=args.show_chunks,
        )
        elapsed = time.time() - start

        retrieval_context = recorder.context_texts(
            max_contexts=args.max_contexts,
            max_context_chars=args.max_context_chars,
        )
        expected_output = build_expected_output(case)
        test_case = LLMTestCase(
            input=build_case_input(case),
            actual_output=answer,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        metric_results = measure_metrics(metrics, test_case)
        metric_payload = {}
        for name, result in metric_results.items():
            metric_payload[name] = {
                "score": result.score,
                "success": result.success,
                "reason": result.reason,
            }
            if result.score is not None:
                summary_scores.setdefault(name, []).append(float(result.score))

        print(
            "Scores: "
            + ", ".join(
                f"{name}={payload['score']:.3f}"
                for name, payload in metric_payload.items()
                if payload["score"] is not None
            )
        )

        rows.append(
            {
                "case_id": case["case_id"],
                "focus": case.get("focus"),
                "elapsed_seconds": elapsed,
                "judge_model": judge_name,
                "metrics": metric_payload,
                "retrieval_context_count": len(retrieval_context),
                "expected_output": expected_output,
                "answer": answer,
            }
        )

    summary = {
        "total_cases": len(cases),
        "judge_model": judge_name,
        "threshold": args.threshold,
        "averages": {
            name: sum(values) / len(values)
            for name, values in sorted(summary_scores.items())
            if values
        },
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepEval LLM-judge evaluation against RAG ground truth."
    )
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--case-id", help="Run one or comma-separated case IDs.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected cases.")
    parser.add_argument("--dry-run", action="store_true", help="Validate ground truth and exit.")
    parser.add_argument("--show-chunks", action="store_true", help="Show retrieved chunks from generation.py.")
    parser.add_argument("--no-scoring-result", action="store_true", help="Do not pass mock_tim1_output as scoring_result.")
    parser.add_argument("--judge-model", help="Model used as DeepEval judge.")
    parser.add_argument("--judge-api-url", help="Override OpenAI-compatible judge API URL.")
    parser.add_argument("--judge-api-key", help="Override judge API bearer token.")
    parser.add_argument(
        "--no-judge-json-mode",
        action="store_true",
        help="Disable JSON-only system instruction for judge calls.",
    )
    parser.add_argument("--judge-temperature", type=float, default=RUNPOD_TEMPERATURE)
    parser.add_argument(
        "--async-metrics",
        action="store_true",
        help="Enable DeepEval async metric calls.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,contextual_precision",
        help=(
            "Comma-separated metrics: faithfulness, answer_relevancy, "
            "contextual_precision."
        ),
    )
    parser.add_argument("--max-contexts", type=int, default=40)
    parser.add_argument("--max-context-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()

    cases = load_jsonl(args.ground_truth)
    cases = select_cases(cases, args.case_id, args.limit)
    if not cases:
        raise SystemExit("No cases selected.")

    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("Duplicate case_id found in selected ground truth.")

    print(f"Ground truth : {args.ground_truth}")
    print(f"Selected     : {len(cases)} case(s)")
    print("Case IDs     : " + ", ".join(case_ids))

    if args.dry_run:
        print("Dry run OK. No RAG or DeepEval calls made.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"deepeval_{timestamp}.jsonl"
    summary_path = args.output_dir / f"deepeval_{timestamp}_summary.json"

    rows, summary = run_cases(args, cases)
    summary.update(
        {
            "timestamp": timestamp,
            "ground_truth": str(args.ground_truth),
            "result_path": str(result_path),
            "case_ids": case_ids,
        }
    )

    write_jsonl(result_path, rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDeepEval complete.")
    print(f"Results : {result_path}")
    print(f"Summary : {summary_path}")
    print("Averages:")
    for name, value in summary.get("averages", {}).items():
        print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
