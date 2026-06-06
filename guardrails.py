import re
import json
import logging

# Custom imports
from generation import PROGRAM_LABELS
from retrieval import RetrievalResult
from helpers import (
    parse_profile_signals,
    normalize_program_name,
    normalize_tim1_output,
    source_ref_for_program,
)

logger = logging.getLogger(__name__)


def tim1_is_layak(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    status = str(item.get("status_kelayakan") or "").upper()
    label = item.get("label")
    return ("LAYAK" in status and "TIDAK" not in status) or label == 1


def enforce_program_eligibility_rules(parsed: dict, profil_warga: str) -> dict:
    """
    Guardrail deterministik setelah LLM.
    LLM tidak boleh meloloskan program yang melanggar hard rule juknis.
    """
    if not isinstance(parsed, dict):
        return parsed

    signals = parse_profile_signals(profil_warga)
    age = signals.get("umur")
    skor_pkh = signals.get("skor_pkh_plus")

    data = parsed.copy()
    rekomendasi_raw = data.get("rekomendasi") if isinstance(data.get("rekomendasi"), list) else []
    tidak_sesuai_raw = (
        data.get("program_tidak_sesuai")
        if isinstance(data.get("program_tidak_sesuai"), list)
        else []
    )

    rekomendasi: list[dict] = []
    tidak_sesuai: list[dict] = [
        item.copy() for item in tidak_sesuai_raw if isinstance(item, dict)
    ]

    def add_tidak_sesuai(program_name: str, alasan: str):
        canonical = normalize_program_name(program_name)
        for item in tidak_sesuai:
            if normalize_program_name(str(item.get("nama_program") or "")) == canonical:
                item["nama_program"] = canonical
                item["status"] = "TIDAK_ELIGIBLE"
                item["alasan"] = alasan
                return
        tidak_sesuai.append({
            "nama_program": canonical,
            "status": "TIDAK_ELIGIBLE",
            "alasan": alasan,
        })

    for item in rekomendasi_raw:
        if not isinstance(item, dict):
            continue
        current = item.copy()
        canonical = normalize_program_name(str(current.get("nama_program") or ""))
        current["nama_program"] = canonical

        is_pkh_plus = canonical == "PKH Plus (Lanjut Usia 70+)"
        if is_pkh_plus and age is not None and age < 70:
            alasan = (
                f"Tidak memenuhi hard rule PKH Plus: usia warga {age} tahun, "
                "sedangkan sasaran PKH Plus adalah lanjut usia 70 tahun ke atas."
            )
            if skor_pkh is not None:
                alasan += f" Skor PKH Plus dari profil: {skor_pkh}."
            add_tidak_sesuai(canonical, alasan)
            continue

        if is_pkh_plus and skor_pkh is not None and float(skor_pkh) <= 0.05:
            add_tidak_sesuai(
                canonical,
                f"Tidak direkomendasikan karena skor PKH Plus dari profil adalah {skor_pkh}, "
                "di bawah ambang prioritas."
            )
            continue

        rekomendasi.append(current)

    for idx, item in enumerate(rekomendasi, 1):
        item["rank"] = idx

    data["rekomendasi"] = rekomendasi
    data["program_tidak_sesuai"] = tidak_sesuai
    return data


def build_fallback_generation(
    profil_warga: str,
    results: list[RetrievalResult],
) -> dict:
    tim1 = {}
    laporan = {}
    profil = {}
    analisis = {}
    parameter = {}
    kesimpulan = {}
    skor = {}
    profile_signals = parse_profile_signals(profil_warga)

    umur = profil.get("umur") or profile_signals.get("umur")
    desil = parameter.get("desil_nasional") or profile_signals.get("desil_nasional")
    status_dtsen = (
        profil.get("status_dtsen")
        or parameter.get("status_dtsekolah")
        or profile_signals.get("status_dtsen")
    )
    wilayah = profil.get("wilayah")
    if isinstance(wilayah, dict):
        wilayah_text = ", ".join(str(v) for v in wilayah.values() if v)
    else:
        wilayah_text = str(wilayah or profile_signals.get("lokasi") or "")

    ringkasan_parts = []
    if umur is not None:
        ringkasan_parts.append(f"umur {umur} tahun")
    if desil is not None:
        ringkasan_parts.append(f"desil nasional {desil}")
    if status_dtsen:
        ringkasan_parts.append(str(status_dtsen))
    if wilayah_text:
        ringkasan_parts.append(wilayah_text)
    if analisis.get("disabilitas_fungsi"):
        ringkasan_parts.append(str(analisis["disabilitas_fungsi"]))
    ringkasan = (
        "Profil warga: " + "; ".join(ringkasan_parts)
        if ringkasan_parts
        else profil_warga[:500]
    )

    program_configs = [
        (
            "pkh_plus",
            "PKH Plus (Lanjut Usia 70+)",
            "JUKNIS PKH PLUS 2026.pdf",
            analisis.get("sintesis_pkh_plus"),
            skor.get("skor_pkh_plus", profile_signals.get("skor_pkh_plus")),
            {
                "nominal_bantuan": "Mengacu JUKNIS PKH Plus 2026",
                "frekuensi": "sesuai tahapan penyaluran dalam juknis",
                "sasaran": "lansia 70 tahun ke atas yang memenuhi kriteria DTSEN/desil dan administrasi kependudukan Jawa Timur",
                "syarat_dokumen": ["KTP", "KK", "NIK"],
                "mekanisme": "verifikasi/pemutakhiran data dan penyaluran sesuai petunjuk teknis PKH Plus",
            },
        ),
        (
            "aspd",
            "Asistensi Sosial Penyandang Disabilitas (ASPD)",
            "Juklak ASPD Tahun 202620260225_12303533_01.pdf",
            analisis.get("sintesis_aspd"),
            skor.get("skor_aspd", profile_signals.get("skor_aspd")),
            {
                "nominal_bantuan": "Mengacu Juklak ASPD Tahun 2026",
                "frekuensi": "sesuai tahapan penyaluran dalam juklak",
                "sasaran": "penyandang disabilitas yang memenuhi kriteria usia, domisili, desil/prioritas, dan verifikasi lapangan",
                "syarat_dokumen": ["KTP", "KK", "NIK", "dokumen pendukung disabilitas/verifikasi"],
                "mekanisme": "verifikasi data penerima, penetapan, dan penyaluran melalui mekanisme juklak ASPD",
            },
        ),
    ]

    rekomendasi = []
    tidak_sesuai = []
    rank = 1
    for key, program_name, source, sintesis, score, spec in program_configs:
        kes = kesimpulan.get(key)
        inferred_layak = False
        if key == "pkh_plus":
            inferred_layak = (
                umur is not None and umur >= 70
                and desil is not None and desil <= 4
                and status_dtsen and "aktif" in str(status_dtsen).lower()
            )
            if not sintesis and inferred_layak:
                sintesis = (
                    f"Warga berusia {umur} tahun, memenuhi batas lansia 70 tahun ke atas; "
                    f"desil nasional {desil} masuk prioritas 1-4; status DTSEN aktif."
                )
        elif key == "aspd":
            inferred_layak = (
                profile_signals.get("has_disability")
                and umur is not None and umur <= 60
                and desil is not None and desil <= 5
            )
            if not sintesis and inferred_layak:
                sintesis = (
                    f"Warga memiliki indikasi hambatan fungsi/disabilitas, usia {umur} tahun "
                    f"masuk rentang ASPD, dan desil nasional {desil} masuk prioritas."
                )

        alasan = str(sintesis or "Tidak ada sintesis Tim 1 yang tersedia.")
        if score is not None:
            alasan = f"{alasan} Skor Tim 1: {score}."

        if tim1_is_layak(kes) or inferred_layak:
            rekomendasi.append({
                "rank": rank,
                "nama_program": program_name,
                "status": "ELIGIBLE",
                "sumber": source_ref_for_program(results, source),
                "alasan_kelayakan": alasan,
                "spesifikasi": spec,
            })
            rank += 1
        else:
            tidak_sesuai.append({
                "nama_program": program_name,
                "status": "TIDAK_ELIGIBLE",
                "alasan": alasan,
            })

    for program_name in PROGRAM_LABELS.values():
        if program_name not in [r["nama_program"] for r in rekomendasi] and program_name not in [r["nama_program"] for r in tidak_sesuai]:
            tidak_sesuai.append({
                "nama_program": program_name,
                "status": "TIDAK_ELIGIBLE",
                "alasan": "Tidak ada indikator profil dan hasil Tim 1 yang menunjukkan kecocokan utama untuk program ini.",
            })

    return {
        "ringkasan_profil": ringkasan,
        "rekomendasi": rekomendasi,
        "program_tidak_sesuai": tidak_sesuai,
    }
