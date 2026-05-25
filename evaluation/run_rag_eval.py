"""
Run lightweight RAG evaluation against evaluation/ground_truth_rag_cases.jsonl.

This script intentionally uses deterministic string/regex checks first, not an
LLM judge. It gives a cheap baseline before changing retrieval/generation.

Example:
    python evaluation/run_rag_eval.py --dry-run
    python evaluation/run_rag_eval.py --limit 3
    python evaluation/run_rag_eval.py --case-id GT_ASPD_001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GROUND_TRUTH_DEFAULT = Path(__file__).resolve().parent / "ground_truth_rag_cases.jsonl"
RESULTS_DIR_DEFAULT = Path(__file__).resolve().parent / "results"
NOMINAL_SOURCE_DEFAULT = REPO_ROOT / "chunked_data" / "juknis_extracted_normalized.jsonl"


MAIN_PROGRAM_ALIASES = {
    "ASPD": [
        "aspd",
        "asistensi sosial penyandang disabilitas",
    ],
    "Kemiskinan Ekstrem": [
        "kemiskinan ekstrem",
        "penanganan kemiskinan ekstrem",
    ],
    "PKH Plus": [
        "pkh plus",
    ],
    "KIP KPM Jawara": [
        "kip kpm jawara",
        "kip kpm",
        "kpm jawara",
    ],
    "KIP PPKS Jawara": [
        "kip ppks jawara",
        "kip ppks",
        "ppks jawara",
    ],
    "KIP Putri Jawara": [
        "kip putri jawara",
        "kip putri",
        "putri jawara",
    ],
}


STATUS_ORDER = ("eligible", "mungkin_eligible", "tidak_eligible")
DETECTION_STATUS_PRIORITY = ("tidak_eligible", "mungkin_eligible", "eligible")


STATUS_LABELS = {
    "eligible": "eligible",
    "mungkin eligible": "mungkin_eligible",
    "tidak eligible": "tidak_eligible",
    "non eligible": "tidak_eligible",
    "non-eligible": "tidak_eligible",
}


@dataclass
class CaseScore:
    case_id: str
    overall_score: float
    main_status_score: float
    evidence_score: float
    output_contract_score: float
    nominal_score: float
    expected_statuses: dict[str, str]
    detected_statuses: dict[str, str]
    failed_checks: list[str]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("_", " ")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def expected_status_map(case: dict[str, Any]) -> dict[str, str]:
    expected = {}
    main = case["expected_main_programs"]
    for status in STATUS_ORDER:
        for program in main.get(status, []):
            expected[program] = status
    return expected


def find_alias_positions(answer_norm: str, aliases: list[str]) -> list[int]:
    positions = []
    for alias in aliases:
        alias_norm = normalize(alias)
        start = 0
        while True:
            idx = answer_norm.find(alias_norm, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(alias_norm)
    return sorted(set(positions))


def canonical_status(label: str) -> str:
    return STATUS_LABELS.get(normalize(label), "unknown")


def canonical_program_from_text(text: str) -> str | None:
    text_norm = normalize(text)
    # Prefer longer aliases first so "kip kpm jawara" wins over shorter fragments.
    candidates = []
    for program, aliases in MAIN_PROGRAM_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize(alias)
            if alias_norm in text_norm:
                candidates.append((len(alias_norm), program))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def extract_status_sections(answer: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Extract statuses from explicit markdown headings such as:
    ### Rank 1: ASPD — STATUS: ELIGIBLE ✅

    This avoids the old false positive where an explanation containing
    "tidak memenuhi" overrode a header that said "MUNGKIN ELIGIBLE".
    """
    heading_re = re.compile(
        r"(?im)^#{2,4}\s*(?P<title>.+?)\s*(?:—|-)\s*STATUS\s*:\s*"
        r"(?P<status>TIDAK\s+ELIGIBLE|MUNGKIN\s+ELIGIBLE|ELIGIBLE)\b.*$"
    )
    heading_re = re.compile(
        r"(?im)^#{2,4}\s*(?P<title>.+?)\s*(?:â€”|—|–|-|:)\s*STATUS\s*:\s*"
        r"(?P<status>TIDAK\s+ELIGIBLE|NON[-\s]+ELIGIBLE|MUNGKIN\s+ELIGIBLE|ELIGIBLE)\b.*$"
    )
    matches = list(heading_re.finditer(answer))
    statuses: dict[str, str] = {}
    sections: dict[str, str] = {}

    for idx, match in enumerate(matches):
        title = match.group("title")
        program = canonical_program_from_text(title)
        if not program:
            continue
        statuses[program] = canonical_status(match.group("status"))
        section_start = match.start()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(answer)
        sections[program] = answer[section_start:section_end]

    # Fallback for nonstandard outputs such as:
    # "## Program 3: PKH Plus" followed by "**Keputusan:** **Eligible**".
    heading_fallback_re = re.compile(r"(?im)^##(?!#)\s*(?P<title>.+?)\s*$")
    fallback_matches = list(heading_fallback_re.finditer(answer))
    for idx, match in enumerate(fallback_matches):
        title = match.group("title")
        program = canonical_program_from_text(title)
        if not program or program in statuses:
            continue

        section_start = match.start()
        section_end = fallback_matches[idx + 1].start() if idx + 1 < len(fallback_matches) else len(answer)
        section = answer[section_start:section_end]
        section_norm = normalize(section)
        decision_match = re.search(
            r"\b(?:status|keputusan)\s*:?\s*\**\s*"
            r"(tidak\s+eligible|non[-\s]+eligible|mungkin\s+eligible|eligible)\b",
            section_norm,
        )
        status = canonical_status(decision_match.group(1)) if decision_match else status_from_window(section_norm)
        if status != "unknown":
            statuses[program] = status
            sections[program] = section

    return statuses, sections


def status_from_window(window: str) -> str:
    # Order matters: "tidak eligible" contains "eligible".
    if re.search(r"\bnon[-\s]+eligible\b", window):
        return "tidak_eligible"
    if re.search(r"\btidak\s+eligible\b", window) or "tidak memenuhi" in window or "tidak sesuai" in window:
        return "tidak_eligible"
    if re.search(r"\bmungkin\s+eligible\b", window) or "perlu verifikasi" in window or "belum dapat dipastikan" in window:
        return "mungkin_eligible"
    if re.search(r"\beligible\b", window) or "memenuhi syarat" in window or "layak" in window:
        return "eligible"
    return "unknown"


def detect_program_statuses(answer: str) -> dict[str, str]:
    header_statuses, _ = extract_status_sections(answer)
    answer_norm = normalize(answer)
    detected = {}

    for program, aliases in MAIN_PROGRAM_ALIASES.items():
        if program in header_statuses:
            detected[program] = header_statuses[program]
            continue

        positions = find_alias_positions(answer_norm, aliases)
        if not positions:
            detected[program] = "missing"
            continue

        window_statuses = []
        for idx in positions:
            start = max(0, idx - 220)
            end = min(len(answer_norm), idx + 420)
            window_statuses.append(status_from_window(answer_norm[start:end]))

        # Prefer concrete statuses over unknown.
        for status in DETECTION_STATUS_PRIORITY:
            if status in window_statuses:
                detected[program] = status
                break
        else:
            detected[program] = "unknown"

    return detected


def page_reference_present(answer_norm: str, page_number: int) -> bool:
    page = re.escape(str(page_number))
    patterns = [
        rf"\bhal\.?\s*{page}\b",
        rf"\bhalaman\s*{page}\b",
        rf"\bpage\s*{page}\b",
    ]
    if any(re.search(pattern, answer_norm) for pattern in patterns):
        return True

    # Also accept compact multi-page citations that humans understand:
    # "Hal. 13, 14", "Hal. 13 dan 14", "Hal. 13-14".
    citation_re = re.compile(
        r"\b(?:hal\.?|halaman|page)\s*"
        r"(?P<pages>\d+(?:\s*(?:,|dan|&|-|s/d|sd)\s*\d+)*)"
    )
    for match in citation_re.finditer(answer_norm):
        pages_text = match.group("pages")
        nums = [int(num) for num in re.findall(r"\d+", pages_text)]
        if int(page_number) in nums:
            return True

        range_match = re.search(r"(\d+)\s*(?:-|s/d|sd)\s*(\d+)", pages_text)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= int(page_number) <= end:
                return True

    return False


def evaluate_evidence(case: dict[str, Any], answer: str) -> tuple[float, list[dict[str, Any]], list[str]]:
    answer_norm = normalize(answer)
    requirements = case.get("expected_evidence_requirements", [])
    if not requirements:
        return 1.0, [], []

    details = []
    failed = []
    passed = 0

    for req in requirements:
        source = normalize(req["source_contains"])
        page = int(req["page_number"])
        source_ok = source in answer_norm
        page_ok = page_reference_present(answer_norm, page)
        ok = source_ok and page_ok
        if ok:
            passed += 1
        else:
            failed.append(f"evidence:{req.get('program', '?')}:{req['source_contains']}:hal{page}")
        details.append({
            "program": req.get("program"),
            "source_contains": req["source_contains"],
            "page_number": page,
            "source_ok": source_ok,
            "page_ok": page_ok,
            "passed": ok,
        })

    return passed / len(requirements), details, failed


def extract_status_heading_programs(answer: str) -> tuple[dict[str, int], list[str]]:
    heading_re = re.compile(
        r"(?im)^#{2,4}\s*(?P<title>.+?)\s*(?:â€”|—|–|-|:)\s*STATUS\s*:\s*"
        r"(?P<status>TIDAK\s+ELIGIBLE|NON[-\s]+ELIGIBLE|MUNGKIN\s+ELIGIBLE|ELIGIBLE)\b.*$"
    )
    counts: dict[str, int] = {}
    unknown_headings = []

    for match in heading_re.finditer(answer):
        title = match.group("title")
        program = canonical_program_from_text(title)
        if program:
            counts[program] = counts.get(program, 0) + 1
        else:
            unknown_headings.append(title.strip())

    return counts, unknown_headings


def evaluate_output_contract(case: dict[str, Any], answer: str) -> tuple[float, dict[str, Any], list[str]]:
    expected_programs = list(expected_status_map(case).keys())
    counts, unknown_headings = extract_status_heading_programs(answer)
    answer_norm = normalize(answer)
    forbidden_extra_section = (
        "rekomendasi bantuan tambahan" in answer_norm
        or "rekomendasi tambahan" in answer_norm
        or "bantuan lain" in answer_norm
        or "rekomendasi tindak lanjut" in answer_norm
        or "catatan untuk petugas" in answer_norm
    )

    failed = []
    passed = 0
    total = len(expected_programs) + 2

    program_counts = {}
    for program in expected_programs:
        count = counts.get(program, 0)
        program_counts[program] = count
        if count == 1:
            passed += 1
        elif count == 0:
            failed.append(f"output_contract:{program}:missing_heading")
        else:
            failed.append(f"output_contract:{program}:duplicate_heading_count={count}")

    if unknown_headings:
        failed.append("output_contract:unknown_status_heading")
    else:
        passed += 1

    if forbidden_extra_section:
        failed.append("output_contract:forbidden_section_present")
    else:
        passed += 1

    details = {
        "required_programs": expected_programs,
        "program_heading_counts": program_counts,
        "unknown_status_headings": unknown_headings,
        "forbidden_extra_section": forbidden_extra_section,
    }
    return passed / total if total else 1.0, details, failed


def evaluate_notes(answer: str) -> tuple[float, dict[str, bool], list[str]]:
    answer_norm = normalize(answer)
    petugas_ok = "petugas" in answer_norm
    warga_ok = "warga" in answer_norm or "penerima" in answer_norm
    score = (int(petugas_ok) + int(warga_ok)) / 2
    failed = []
    if not petugas_ok:
        failed.append("notes:petugas_missing")
    if not warga_ok:
        failed.append("notes:warga_missing")
    return score, {"petugas": petugas_ok, "warga": warga_ok}, failed


RUPIAH_RE = re.compile(r"\brp\.?\s*([0-9][0-9.,\s]*)(?:,-|,00)?", flags=re.IGNORECASE)
NOMINAL_CONTEXT_RE = re.compile(
    r"\b(bantuan|bansos|besaran|nominal|sebesar|senilai|per orang|per tahap|"
    r"tahap|disalurkan|diterima|penerima manfaat)\b",
    flags=re.IGNORECASE,
)
NON_BANTUAN_AMOUNT_RE = re.compile(
    r"\b(materai|bea\s+materai)\b.{0,50}\brp\b|\brp\b.{0,50}\b(materai|bea\s+materai)\b",
    flags=re.IGNORECASE,
)


def canonical_amount(amount: str) -> str:
    return re.sub(r"\D", "", amount)


def display_amount(amount: str) -> str:
    compact = canonical_amount(amount)
    if compact.isdigit():
        return f"{int(compact):,}".replace(",", ".")
    return amount


def metadata_tags(metadata: dict[str, Any]) -> set[str]:
    tags = metadata.get("tipe_konten", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []
    primary = metadata.get("tipe_konten_primer")
    if primary:
        tags.append(primary)
    return {str(tag) for tag in tags}


def is_bansos_nominal_context(context: str) -> bool:
    if NON_BANTUAN_AMOUNT_RE.search(context):
        return False
    return bool(NOMINAL_CONTEXT_RE.search(context))


def extract_rupiah_mentions(text: str) -> list[dict[str, Any]]:
    mentions = []
    for match in RUPIAH_RE.finditer(text):
        raw = match.group(1)
        normalized = re.sub(r"\s+", "", raw).strip(".,")
        canonical = canonical_amount(normalized)
        if not canonical:
            continue

        context_start = max(0, match.start() - 140)
        context_end = min(len(text), match.end() + 180)
        context = re.sub(r"\s+", " ", text[context_start:context_end]).strip()
        mentions.append({
            "amount": normalized,
            "canonical": canonical,
            "display": display_amount(normalized),
            "context": context,
        })
    return mentions


def extract_rupiah_amounts(text: str) -> list[str]:
    return [mention["amount"] for mention in extract_rupiah_mentions(text)]


def build_nominal_catalog(path: Path) -> dict[str, Any]:
    """
    Build per-program nominal facts from cleaned Juknis JSONL.

    The evaluator should follow data, not code. If a new Juknis changes a
    nominal, regenerate the normalized JSONL and this catalog changes with it.
    """
    if not path.exists():
        return {}

    catalog: dict[str, Any] = {}
    for row in load_jsonl(path):
        metadata = row.get("metadata") or {}
        program = metadata.get("nama_bansos")
        if not program:
            continue

        tags = metadata_tags(metadata)
        text = row.get("text", "")
        if "nominal_bantuan" not in tags and not NOMINAL_CONTEXT_RE.search(text):
            continue

        mentions = [
            mention for mention in extract_rupiah_mentions(text)
            if is_bansos_nominal_context(mention["context"])
        ]
        if not mentions:
            continue

        entry = catalog.setdefault(program, {
            "source_path": str(path),
            "amounts": {},
        })
        for mention in mentions:
            amount_key = mention["canonical"]
            amount_entry = entry["amounts"].setdefault(amount_key, {
                "display": mention["display"],
                "evidence": [],
            })
            evidence = amount_entry["evidence"]
            evidence_key = (
                metadata.get("sumber"),
                metadata.get("page_number"),
                mention["context"],
            )
            if len(evidence) < 5 and evidence_key not in {
                (item.get("sumber"), item.get("page_number"), item.get("context"))
                for item in evidence
            }:
                evidence.append({
                    "sumber": metadata.get("sumber"),
                    "page_number": metadata.get("page_number"),
                    "tipe_konten_primer": metadata.get("tipe_konten_primer"),
                    "context": mention["context"],
                })

    return catalog


def summarize_nominal_catalog(catalog: dict[str, Any]) -> str:
    if not catalog:
        return "0 program"
    program_count = len(catalog)
    amount_count = sum(len(entry.get("amounts", {})) for entry in catalog.values())
    return f"{program_count} program, {amount_count} nominal"


def evaluate_nominals(
    case: dict[str, Any],
    answer: str,
    nominal_catalog: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any], list[str]]:
    if nominal_catalog is None:
        nominal_catalog = build_nominal_catalog(NOMINAL_SOURCE_DEFAULT)
    if not nominal_catalog:
        return 1.0, {"catalog_missing": True}, []

    statuses, sections = extract_status_sections(answer)
    details: dict[str, Any] = {}
    failed = []
    checks = 0
    passed = 0

    programs_to_check = set(sections) | set(expected_status_map(case))
    for program in sorted(programs_to_check):
        section = sections.get(program, "")
        if not section:
            continue

        # Only evaluate nominal for programs that the model writes as ranked/eligible-ish.
        status = statuses.get(program)
        if status not in {"eligible", "mungkin_eligible"}:
            continue

        checks += 1
        section_norm = normalize(section)
        mentions = extract_rupiah_mentions(section)
        amounts = [mention["amount"] for mention in mentions]
        found_keys = {mention["canonical"] for mention in mentions}

        catalog_entry = nominal_catalog.get(program, {})
        catalog_amounts = catalog_entry.get("amounts", {})
        allowed_keys = set(catalog_amounts)
        allowed_display = [
            catalog_amounts[key]["display"]
            for key in sorted(allowed_keys, key=lambda value: int(value) if value.isdigit() else value)
        ]

        unsupported = [
            mention for mention in mentions
            if mention["canonical"] not in allowed_keys
        ]
        missing_catalog_amounts = [
            catalog_amounts[key]["display"]
            for key in sorted(allowed_keys, key=lambda value: int(value) if value.isdigit() else value)
            if key not in found_keys
        ]
        claims_not_in_doc = (
            "nominal tidak tersebut" in section_norm
            or "nominal tidak disebut" in section_norm
            or "nominal tidak ditemukan" in section_norm
        )

        if allowed_keys:
            matched_keys = found_keys & allowed_keys
            completeness = len(matched_keys) / len(allowed_keys)
            ok = not unsupported and not claims_not_in_doc and not missing_catalog_amounts
            program_score = 1.0 if ok else (
                0.0 if unsupported or claims_not_in_doc else completeness
            )
        else:
            # If the source JSONL has no nominal for this program, a generated Rp
            # value is treated as unsupported, but saying no nominal is acceptable.
            ok = not unsupported
            completeness = 1.0 if ok else 0.0
            program_score = completeness

        passed += program_score
        details[program] = {
            "status": status,
            "amounts_found": amounts,
            "catalog_amounts": allowed_display,
            "catalog_source": catalog_entry.get("source_path"),
            "catalog_evidence": {
                catalog_amounts[key]["display"]: catalog_amounts[key].get("evidence", [])
                for key in allowed_keys
            },
            "unsupported_amounts": [mention["amount"] for mention in unsupported],
            "missing_catalog_amounts": missing_catalog_amounts,
            "claims_nominal_not_in_doc": claims_not_in_doc,
            "matched_catalog_ratio": completeness,
            "passed": ok,
        }

        for mention in unsupported:
            failed.append(f"nominal:{program}:unsupported_rp_{mention['amount']}")
        for amount in missing_catalog_amounts:
            failed.append(f"nominal:{program}:missing_catalog_rp_{amount}")
        if claims_not_in_doc and allowed_keys:
            failed.append(f"nominal:{program}:contradiction_nominal_not_in_doc")

    if checks == 0:
        return 1.0, details, failed
    return passed / checks, details, failed


def evaluate_case(
    case: dict[str, Any],
    answer: str,
    nominal_catalog: dict[str, Any] | None = None,
) -> tuple[CaseScore, dict[str, Any]]:
    expected = expected_status_map(case)
    detected = detect_program_statuses(answer)

    status_checks = {}
    status_passed = 0
    for program, expected_status in expected.items():
        detected_status = detected.get(program, "missing")
        ok = detected_status == expected_status
        status_checks[program] = {
            "expected": expected_status,
            "detected": detected_status,
            "passed": ok,
        }
        status_passed += int(ok)

    main_status_score = status_passed / len(expected) if expected else 1.0
    evidence_score, evidence_details, evidence_failed = evaluate_evidence(case, answer)
    output_contract_score, output_contract_details, output_contract_failed = evaluate_output_contract(case, answer)
    nominal_score, nominal_details, nominal_failed = evaluate_nominals(case, answer, nominal_catalog)

    failed_checks = []
    failed_checks.extend(
        f"status:{program}:expected={item['expected']}:detected={item['detected']}"
        for program, item in status_checks.items()
        if not item["passed"]
    )
    failed_checks.extend(evidence_failed)
    failed_checks.extend(output_contract_failed)
    failed_checks.extend(nominal_failed)

    overall = (
        main_status_score * 0.40
        + evidence_score * 0.30
        + output_contract_score * 0.20
        + nominal_score * 0.10
    )

    score = CaseScore(
        case_id=case["case_id"],
        overall_score=overall,
        main_status_score=main_status_score,
        evidence_score=evidence_score,
        output_contract_score=output_contract_score,
        nominal_score=nominal_score,
        expected_statuses=expected,
        detected_statuses=detected,
        failed_checks=failed_checks,
    )

    detail = {
        "status_checks": status_checks,
        "evidence_checks": evidence_details,
        "output_contract_checks": output_contract_details,
        "nominal_checks": nominal_details,
    }
    return score, detail


def build_scoring_result(case: dict[str, Any]) -> str:
    mock = case.get("mock_tim1_output", {})
    return json.dumps(mock, ensure_ascii=False, indent=2)


def select_cases(cases: list[dict[str, Any]], case_id: str | None, limit: int | None) -> list[dict[str, Any]]:
    selected = cases
    if case_id:
        wanted = {part.strip() for part in case_id.split(",") if part.strip()}
        selected = [case for case in selected if case["case_id"] in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected


def summarize(scores: list[CaseScore]) -> dict[str, Any]:
    if not scores:
        return {
            "total_cases": 0,
            "averages": {},
        }

    def avg(attr: str) -> float:
        return sum(getattr(score, attr) for score in scores) / len(scores)

    return {
        "total_cases": len(scores),
        "averages": {
            "overall_score": avg("overall_score"),
            "main_status_score": avg("main_status_score"),
            "evidence_score": avg("evidence_score"),
            "output_contract_score": avg("output_contract_score"),
            "nominal_score": avg("nominal_score"),
        },
        "failed_cases": [
            {
                "case_id": score.case_id,
                "overall_score": score.overall_score,
                "failed_checks": score.failed_checks,
            }
            for score in scores
            if score.failed_checks
        ],
    }


def run_rag_cases(args: argparse.Namespace, cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[CaseScore]]:
    from generation import RAGGenerator

    nominal_catalog = build_nominal_catalog(args.nominal_source)
    print(f"Nominal source: {args.nominal_source} ({summarize_nominal_catalog(nominal_catalog)})")

    rag = RAGGenerator()
    rows = []
    scores = []

    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] Running {case['case_id']} - {case.get('focus', '')}")
        start = time.time()
        scoring_result = "" if args.no_scoring_result else build_scoring_result(case)
        answer = rag.recommend(
            case["profil_warga"],
            scoring_result=scoring_result,
            stream=False,
            show_chunks=args.show_chunks,
        )
        elapsed = time.time() - start

        score, detail = evaluate_case(case, answer, nominal_catalog)
        scores.append(score)
        rows.append({
            "case_id": case["case_id"],
            "focus": case.get("focus"),
            "elapsed_seconds": elapsed,
            "scores": {
                "overall_score": score.overall_score,
                "main_status_score": score.main_status_score,
                "evidence_score": score.evidence_score,
                "output_contract_score": score.output_contract_score,
                "nominal_score": score.nominal_score,
            },
            "failed_checks": score.failed_checks,
            "detected_statuses": score.detected_statuses,
            "expected_statuses": score.expected_statuses,
            "details": detail,
            "answer": answer,
        })

        print(
            "Score "
            f"overall={score.overall_score:.2f} "
            f"status={score.main_status_score:.2f} "
            f"evidence={score.evidence_score:.2f} "
            f"contract={score.output_contract_score:.2f} "
            f"nominal={score.nominal_score:.2f}"
        )
        if score.failed_checks:
            print("Failed checks:")
            for failed in score.failed_checks[:8]:
                print(f"  - {failed}")
            if len(score.failed_checks) > 8:
                print(f"  ... {len(score.failed_checks) - 8} more")

    return rows, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG evaluation against seed ground truth.")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--case-id", help="Run one or comma-separated case IDs.")
    parser.add_argument("--limit", type=int, help="Run only the first N selected cases.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate ground truth and exit.")
    parser.add_argument("--show-chunks", action="store_true", help="Show retrieved chunks from generation.py.")
    parser.add_argument("--no-scoring-result", action="store_true", help="Do not pass mock_tim1_output as scoring_result.")
    parser.add_argument(
        "--nominal-source",
        type=Path,
        default=NOMINAL_SOURCE_DEFAULT,
        help="Cleaned Juknis JSONL used to derive nominal checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_jsonl(args.ground_truth)
    cases = select_cases(cases, args.case_id, args.limit)

    if not cases:
        raise SystemExit("No cases selected.")

    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("Duplicate case_id found in selected ground truth.")

    print(f"Ground truth: {args.ground_truth}")
    print(f"Selected cases: {len(cases)}")
    print("Case IDs:", ", ".join(case_ids))

    if args.dry_run:
        print("Dry run OK. No RAG calls made.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"rag_eval_{timestamp}.jsonl"
    summary_path = args.output_dir / f"rag_eval_{timestamp}_summary.json"

    result_rows, scores = run_rag_cases(args, cases)
    summary = summarize(scores)
    summary.update({
        "timestamp": timestamp,
        "ground_truth": str(args.ground_truth),
        "nominal_source": str(args.nominal_source),
        "result_path": str(result_path),
        "case_ids": case_ids,
    })

    write_jsonl(result_path, result_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nEvaluation complete.")
    print(f"Results : {result_path}")
    print(f"Summary : {summary_path}")
    print("Averages:")
    for key, value in summary["averages"].items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
