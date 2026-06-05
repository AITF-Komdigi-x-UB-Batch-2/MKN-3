import re
import json
import logging
import httpx
from typing import Optional
from fastapi import HTTPException

# Custom config settings
from config import (
    RUNPOD_API_KEY, RUNPOD_MODEL_NAME, RUNPOD_TEMPERATURE, RUNPOD_MAX_TOKENS,
    TIM1_CLASSIFICATION_API_URL, TIM1_GENERATION_API_URL, TIM1_API_TIMEOUT_S
)

logger = logging.getLogger(__name__)

# Global reference to local LLM, to be populated by webservice.py during lifespan startup
local_llm = None


def parse_llm_json(raw: str) -> dict:
    """
    Ekstrak dan parse JSON dari output LLM.
    Handle kasus LLM membungkus JSON dengan ```json``` atau teks tambahan.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Cari blok JSON pertama
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("⚠️ LLM tidak menghasilkan JSON valid. Fallback ke raw.")
    return {"_raw": raw, "_parse_error": True}


def invoke_llm(prompt: str) -> dict:
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
    try:
        content = api_response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
    except (KeyError, IndexError, TypeError):
        pass

    raise ValueError("Response API generation tidak memiliki choices[0].message.content.")


def call_runpod_chat_api(api_url: str, messages: list[dict]) -> str:
    if not RUNPOD_API_KEY:
        raise RuntimeError("RUNPOD_API_KEY belum terisi di .env.")

    payload = {
        "model": RUNPOD_MODEL_NAME,
        "messages": messages,
        "temperature": RUNPOD_TEMPERATURE,
        "max_tokens": RUNPOD_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
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
    return call_runpod_chat_api(
        TIM1_CLASSIFICATION_API_URL,
        [{"role": "user", "content": profil_warga}],
    )


def call_generation_api(messages: list[dict]) -> dict:
    raw_content = call_runpod_chat_api(TIM1_GENERATION_API_URL, messages)
    return parse_llm_json(raw_content)


def is_placeholder_generation(parsed: dict) -> bool:
    if not isinstance(parsed, dict):
        return True

    placeholder_terms = [
        "rangkuman singkat",
        "nama program",
        "rp x.xxx.xxx",
        "penjelasan mengapa",
        "nama dokumen dan bagian",
        "kriteria penerima sesuai juknis",
        "cara pencairan/penyaluran",
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

    rekomendasi = parsed.get("rekomendasi")
    tidak_sesuai = parsed.get("program_tidak_sesuai")
    if not isinstance(rekomendasi, list) or not isinstance(tidak_sesuai, list):
        return True

    return False


def call_generation_api_checked(messages: list[dict]) -> dict:
    parsed = call_generation_api(messages)
    if not is_placeholder_generation(parsed):
        return parsed

    retry_messages = messages + [
        {
            "role": "user",
            "content": (
                "Output sebelumnya ditolak karena mengulang prompt atau menyalin placeholder schema. "
                "Jawab ulang HANYA dengan JSON final. Isi semua field dengan nilai konkret berdasarkan "
                "profil warga, hasil Tim 1, dan konteks dokumen. Jangan menulis ulang instruksi, "
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
    if parsed.get("_parse_error"):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "LLM tidak menghasilkan JSON valid. Coba ulangi request.",
                "raw_output_preview": parsed.get("_raw", "")[:500],
            }
        )
