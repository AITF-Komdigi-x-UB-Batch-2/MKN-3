import re
import csv
import io
import json
import logging
import httpx
from fastapi import HTTPException

# Custom config settings
from config import (
    RUNPOD_API_KEY, RUNPOD_MODEL_NAME, RUNPOD_TEMPERATURE, RUNPOD_MAX_TOKENS,
    MKN1_GENERATION_ENDPOINT_MODEL, TIM1_API_TIMEOUT_S
)

logger = logging.getLogger(__name__)

# Global reference to local LLM, to be populated by webservice.py during lifespan startup
local_llm = None


# ============================================================
# HELPERS
# ============================================================

def _strip_thinking_tags(raw: str) -> str:
    """
    Hapus blok <think>...</think> dari output model (Qwen3 thinking mode).
    Tag ini muncul sebelum output utama dan dapat merusak TOON/JSON parser.
    Juga menghapus tag yang belum ditutup (model terputus saat masih berpikir).
    """
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=re.IGNORECASE).strip()
    # Hapus juga tag yang belum ditutup (model cut off di tengah thinking)
    cleaned = re.sub(r'<think>[\s\S]*$', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


# ============================================================
# TOON PARSER
# ============================================================

def parse_toon_format(raw: str) -> dict:
    """
    Parses a TOON (Token-oriented Object Notation) string into a dictionary
    matching the schema required by the recommendation pipeline.

    Expected row format (CSV-like with 4 columns):
        kategori, nilai/program, status, "detail/alasan"

    Supported kategori values:
        - ringkasan_profil
        - rekomendasi          → Detail/Alasan: "Rank: X | Dasar Hukum: Y | Alasan: Z"
        - rekomendasi_teknis_bansos
        - program_tidak_sesuai → Detail/Alasan: "Alasan: ..."
    """
    parsed = {
        "ringkasan_profil": "",
        "rekomendasi": [],
        "rekomendasi_teknis_bansos": None,
        "program_tidak_sesuai": []
    }

    # Strip thinking tags (Qwen3 thinking mode) sebelum parsing
    raw = _strip_thinking_tags(raw)

    # Bersihkan markdown wrapper jika ada (misal: ```toon ... ```)
    cleaned = re.sub(r"```(?:toon)?\s*", "", raw).strip().rstrip("`").strip()

    lines = cleaned.splitlines()
    csv_lines = []

    for line in lines:
        l_str = line.strip()
        if not l_str:
            continue
        # Skip header lines:
        # - "Hasil[X]{Kategori,Nilai/Program,Status,Detail/Alasan}:"
        # - Baris mengandung header kolom TOON
        # - Baris komentar
        if (
            l_str.startswith("Hasil[")
            or l_str.startswith("Hasil {")
            or "Kategori,Nilai/Program" in l_str
            or l_str.startswith("#")
        ):
            continue
        # Hanya proses baris yang mengandung minimal 3 koma (4 kolom TOON)
        if l_str.count(",") < 3:
            logger.debug("⚠️ TOON parser skip baris (kurang dari 4 kolom): %s", l_str[:80])
            continue
        csv_lines.append(l_str)

    if not csv_lines:
        logger.warning("⚠️ TOON parser tidak menemukan baris data CSV valid.")
        return {"_raw": raw, "_parse_error": True}

    try:
        reader = csv.reader(io.StringIO("\n".join(csv_lines)))
        for row in reader:
            if len(row) < 4:
                continue
            kategori, nilai_program, status, detail_alasan = [x.strip() for x in row[:4]]

            if kategori == "ringkasan_profil":
                parsed["ringkasan_profil"] = detail_alasan

            elif kategori == "rekomendasi_teknis_bansos":
                parsed["rekomendasi_teknis_bansos"] = (
                    None
                    if not detail_alasan or detail_alasan.lower() in ("null", "-", "")
                    else detail_alasan
                )

            elif kategori == "rekomendasi":
                # Detail/Alasan format: "Rank: <Angka> | Dasar Hukum: <Sumber> | Alasan: <Reasoning>"
                rank = 1
                sumber = ""
                alasan_kelayakan = detail_alasan

                parts = [p.strip() for p in detail_alasan.split("|")]
                for part in parts:
                    if part.lower().startswith("rank:"):
                        try:
                            rank = int(re.search(r"\d+", part).group())
                        except (AttributeError, ValueError):
                            pass
                    elif part.lower().startswith("dasar hukum:"):
                        sumber = part.split(":", 1)[1].strip()
                    elif part.lower().startswith("alasan:"):
                        alasan_kelayakan = part.split(":", 1)[1].strip()

                parsed["rekomendasi"].append({
                    "rank": rank,
                    "nama_program": nilai_program,
                    "status": status,
                    "sumber": sumber,
                    "alasan_kelayakan": alasan_kelayakan,
                })

            elif kategori == "program_tidak_sesuai":
                # Detail/Alasan format: "Alasan: <Reasoning>"
                alasan = detail_alasan
                if detail_alasan.lower().startswith("alasan:"):
                    alasan = detail_alasan.split(":", 1)[1].strip()
                parsed["program_tidak_sesuai"].append({
                    "nama_program": nilai_program,
                    "status": status,
                    "alasan": alasan,
                })

        # Validasi: minimal ada satu kategori yang terisi
        if not parsed["ringkasan_profil"] and not parsed["rekomendasi"] and not parsed["program_tidak_sesuai"]:
            logger.warning("⚠️ TOON parser: tidak ada kategori utama yang terisi setelah parsing.")
            return {"_raw": raw, "_parse_error": True}

        return parsed

    except Exception as e:
        logger.warning("⚠️ Gagal mem-parsing TOON: %s", e)
        return {"_raw": raw, "_parse_error": True}


# ============================================================
# MAIN LLM OUTPUT PARSER
# ============================================================

def parse_llm_json(raw: str) -> dict:
    """
    Ekstrak dan parse output LLM.
    Mendukung format TOON (Token-oriented Object Notation) dan format JSON fallback.
    Urutan pengecekan:
      1. Strip thinking tags
      2. Coba parse sebagai TOON jika ada indikator TOON
      3. Fallback ke JSON parsing
    """
    # Strip thinking tags (Qwen3 thinking mode) sebelum apapun
    raw = _strip_thinking_tags(raw)

    # Cek indikator format TOON terlebih dahulu
    if "Hasil[" in raw or "ringkasan_profil," in raw or (
        "rekomendasi," in raw and "program_tidak_sesuai," in raw
    ):
        parsed_toon = parse_toon_format(raw)
        if not parsed_toon.get("_parse_error"):
            return parsed_toon

    # Fallback ke JSON parsing
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Cari blok JSON pertama dalam output
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("⚠️ LLM tidak menghasilkan format TOON atau JSON valid. Fallback ke raw.")
    return {"_raw": raw, "_parse_error": True}


# ============================================================
# LLM INVOCATION WRAPPERS
# ============================================================

def invoke_llm(prompt: str) -> dict:
    """Invoke local LLM (jika tersedia). Digunakan hanya jika model lokal aktif."""
    global local_llm
    if local_llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM lokal tidak aktif. Gunakan endpoint /recommend yang memakai API RunPod.",
        )
    raw = local_llm.invoke(prompt)
    if not isinstance(raw, str):
        raw = raw.content
    return parse_llm_json(raw)


def extract_chat_content(api_response: dict) -> str:
    """Ekstrak string konten dari response OpenAI-compatible chat API."""
    try:
        content = api_response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
    except (KeyError, IndexError, TypeError):
        pass

    raise ValueError("Response API generation tidak memiliki choices[0].message.content.")


def call_runpod_chat_api(api_url: str, messages: list[dict]) -> str:
    """Kirim chat messages ke RunPod/OpenAI-compatible endpoint dan return string konten."""
    payload = {
        "model": RUNPOD_MODEL_NAME,
        "messages": messages,
        "temperature": RUNPOD_TEMPERATURE,
        "max_tokens": RUNPOD_MAX_TOKENS,
        "extra_body": {
                "enable_thinking": False
        }
    }
    headers = {
        "Content-Type": "application/json",
    }
    if RUNPOD_API_KEY:
        headers["Authorization"] = f"Bearer {RUNPOD_API_KEY}"

    try:
        logger.info("PAYLOAD SENT TO TIM 1: %s", json.dumps(payload, indent=2, ensure_ascii=False))
        with httpx.Client(timeout=TIM1_API_TIMEOUT_S) as client:
            resp = client.post(api_url, json=payload, headers=headers)
            resp.raise_for_status()
            return extract_chat_content(resp.json())
    except httpx.HTTPStatusError as e:
        response = e.response
        body_preview = response.text[:1000] if response is not None else ""
        raise RuntimeError(
            f"Gagal call API Tim 1/RunPod ({api_url}): "
            f"HTTP {response.status_code if response is not None else 'unknown'} "
            f"{response.reason_phrase if response is not None else ''}. "
            f"Response body: {body_preview}"
        ) from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Gagal call API Tim 1/RunPod ({api_url}): {e}") from e


def call_classification_api(profil_warga: str) -> str:
    """Panggil API Tim 1 untuk klasifikasi kelayakan (return raw string)."""
    return call_runpod_chat_api(
        MKN1_GENERATION_ENDPOINT_MODEL,
        [{"role": "user", "content": profil_warga}],
    )


def call_generation_api(messages: list[dict]) -> dict:
    """Panggil API Tim 1 untuk generation dan parse hasilnya (TOON atau JSON)."""
    raw_content = call_runpod_chat_api(MKN1_GENERATION_ENDPOINT_MODEL, messages)
    return parse_llm_json(raw_content)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def is_placeholder_generation(parsed: dict) -> bool:
    """
    Deteksi apakah output LLM adalah placeholder/template yang belum diisi,
    bukan output konkret berdasarkan profil warga.
    """
    if not isinstance(parsed, dict):
        return True

    # Deteksi _parse_error dari TOON/JSON parser
    if parsed.get("_parse_error"):
        return True

    placeholder_terms = [
        "rangkuman singkat",
        "nama program",
        "rp x.xxx.xxx",
        "penjelasan mengapa",
        "nama dokumen dan bagian",
        "kriteria penerima sesuai juknis",
        "cara pencairan/penyaluran",
        "<nama program>",
        "<status>",
        "<angka>",
        "<sumber>",
    ]

    def contains_placeholder(value) -> bool:
        if isinstance(value, str):
            lower = value.lower()
            return any(term in lower for term in placeholder_terms)
        if isinstance(value, dict):
            return any(contains_placeholder(v) for v in value.values())
        if isinstance(value, list):
            return any(contains_placeholder(v) for v in value)
        return False

    if contains_placeholder(parsed):
        return True

    # Validasi struktur wajib: rekomendasi dan program_tidak_sesuai harus berupa list
    rekomendasi = parsed.get("rekomendasi")
    tidak_sesuai = parsed.get("program_tidak_sesuai")
    if not isinstance(rekomendasi, list) or not isinstance(tidak_sesuai, list):
        return True

    return False


def call_generation_api_checked(messages: list[dict]) -> dict:
    """
    Panggil generation API dengan validasi placeholder.
    Jika output pertama adalah placeholder, lakukan satu kali retry dengan instruksi ulang.
    Raise HTTPException 422 jika masih placeholder setelah retry.
    """
    parsed = call_generation_api(messages)
    if not is_placeholder_generation(parsed):
        return parsed

    retry_messages = messages + [
        {
            "role": "user",
            "content": (
                "Output sebelumnya ditolak karena mengulang prompt atau menyalin placeholder schema. "
                "Jawab ulang HANYA dengan format TOON final. Isi semua field dengan nilai konkret berdasarkan "
                "profil warga dan konteks dokumen. Jangan menulis ulang instruksi, "
                "jangan memakai markdown, dan jangan memakai placeholder."
            ),
        }
    ]
    parsed_retry = call_generation_api(retry_messages)
    if is_placeholder_generation(parsed_retry):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Model generation masih menyalin placeholder schema setelah retry.",
                "output_preview": json.dumps(parsed_retry, ensure_ascii=False)[:700],
            },
        )
    return parsed_retry


def raise_if_parse_error(parsed: dict):
    """Raise HTTPException 422 jika parsed dict mengandung _parse_error."""
    if parsed.get("_parse_error"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "LLM tidak menghasilkan format TOON atau JSON valid. Coba ulangi request.",
                "raw_output_preview": parsed.get("_raw", "")[:500],
            }
        )