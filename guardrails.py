import re
import json
import logging

# Custom imports
from generation import PROGRAM_LABELS
from retrieval import RetrievalResult
from helpers import (
    parse_profile_signals,
    normalize_program_name,
    source_ref_for_program,
)

logger = logging.getLogger(__name__)

PKH_PLUS_PROGRAM = "PKH Plus (Lanjut Usia 70+)"
PKH_PLUS_SOURCE = "JUKNIS PKH PLUS 2026.pdf"
ASPD_PROGRAM = "Asistensi Sosial Penyandang Disabilitas (ASPD)"
ASPD_SOURCE = "Juklak ASPD Tahun 202620260225_12303533_01.pdf"


def tim1_is_layak(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    status = str(item.get("status_kelayakan") or "").upper()
    label = item.get("label")
    return ("LAYAK" in status and "TIDAK" not in status) or label == 1


def _status_is_active(value: object) -> bool | None:
    if value is None:
        return None
    text = str(value).lower()
    if not text.strip():
        return None
    if "tidak aktif" in text or "nonaktif" in text or "non aktif" in text:
        return False
    if "aktif" in text:
        return True
    return None


def _evaluate_program_rules(profil_warga: str) -> dict[str, dict]:
    """
    Evaluasi hard rule program dari profil. Hasil ini dipakai setelah LLM,
    sehingga keputusan akhir tidak bergantung penuh pada generasi model.
    """
    signals = parse_profile_signals(profil_warga)
    age = signals.get("umur")
    desil = signals.get("desil_nasional")
    status_active = _status_is_active(signals.get("status_dtsen"))
    has_disability = bool(signals.get("has_disability"))

    pkh_failures = []
    if age is None:
        pkh_failures.append("usia warga tidak terdeteksi")
    elif age < 70:
        pkh_failures.append(
            f"usia warga {age} tahun belum memenuhi syarat PKH Plus 70 tahun ke atas"
        )
    if desil is None:
        pkh_failures.append("desil nasional tidak terdeteksi")
    elif desil > 4:
        pkh_failures.append(
            f"desil nasional {desil} berada di luar sasaran PKH Plus desil 1-4"
        )
    if status_active is False:
        pkh_failures.append("status DTSEN tercatat tidak aktif")

    aspd_failures = []
    if age is None:
        aspd_failures.append("usia warga tidak terdeteksi")
    elif age > 60:
        aspd_failures.append(
            f"usia warga {age} tahun melebihi batas sasaran ASPD maksimal 60 tahun"
        )
    if desil is None:
        aspd_failures.append("desil nasional tidak terdeteksi")
    elif desil > 5:
        aspd_failures.append(
            f"desil nasional {desil} berada di luar prioritas utama ASPD desil 1-5"
        )
    if not has_disability:
        aspd_failures.append(
            "tidak terdeteksi hambatan fungsi berat seperti banyak kesulitan, "
            "sama sekali tidak bisa, tidak mampu, atau membutuhkan bantuan"
        )
    if status_active is False:
        aspd_failures.append("status DTSEN tercatat tidak aktif")

    pkh_eligible = not pkh_failures
    aspd_eligible = not aspd_failures

    pkh_reason = (
        f"Warga memenuhi hard rule PKH Plus: usia {age} tahun, desil nasional {desil}, "
        "dan status DTSEN tidak tercatat bermasalah."
        if pkh_eligible
        else "Tidak memenuhi hard rule PKH Plus: " + "; ".join(pkh_failures) + "."
    )
    aspd_reason = (
        f"Warga memenuhi hard rule ASPD: usia {age} tahun berada dalam rentang maksimal 60 tahun, "
        f"desil nasional {desil} masuk prioritas 1-5, dan profil mencatat hambatan fungsi berat."
        if aspd_eligible
        else "Tidak memenuhi hard rule ASPD: " + "; ".join(aspd_failures) + "."
    )

    return {
        PKH_PLUS_PROGRAM: {
            "eligible": pkh_eligible,
            "source": PKH_PLUS_SOURCE,
            "reason": pkh_reason,
        },
        ASPD_PROGRAM: {
            "eligible": aspd_eligible,
            "source": ASPD_SOURCE,
            "reason": aspd_reason,
        },
    }


def enforce_program_eligibility_rules(parsed: dict, profil_warga: str) -> dict:
    """
    Guardrail deterministik setelah LLM.
    LLM tidak boleh meloloskan program yang melanggar hard rule juknis.
    """
    if not isinstance(parsed, dict):
        return parsed

    rules = _evaluate_program_rules(profil_warga)

    data = parsed.copy()
    rekomendasi_raw = data.get("rekomendasi") if isinstance(data.get("rekomendasi"), list) else []
    tidak_sesuai_raw = (
        data.get("program_tidak_sesuai")
        if isinstance(data.get("program_tidak_sesuai"), list)
        else []
    )

    rekomendasi: list[dict] = []
    tidak_sesuai: list[dict] = []
    allowed_programs = set(PROGRAM_LABELS.values())

    def add_tidak_sesuai(program_name: str, alasan: str):
        canonical = normalize_program_name(program_name)
        if canonical not in allowed_programs:
            return
        rekomendasi[:] = [
            item for item in rekomendasi
            if normalize_program_name(str(item.get("nama_program") or "")) != canonical
        ]
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

    def add_rekomendasi(program_name: str, source: str, alasan: str, status: str = "ELIGIBLE"):
        canonical = normalize_program_name(program_name)
        if canonical not in allowed_programs:
            return
        tidak_sesuai[:] = [
            item for item in tidak_sesuai
            if normalize_program_name(str(item.get("nama_program") or "")) != canonical
        ]
        for item in rekomendasi:
            if normalize_program_name(str(item.get("nama_program") or "")) == canonical:
                item["nama_program"] = canonical
                item["status"] = status
                item["sumber"] = item.get("sumber") or source
                item["alasan_kelayakan"] = alasan
                return
        rekomendasi.append({
            "rank": len(rekomendasi) + 1,
            "nama_program": canonical,
            "status": status,
            "sumber": source,
            "alasan_kelayakan": alasan,
        })

    # Filter program_tidak_sesuai dari LLM agar hanya menyertakan program yang diizinkan
    for item in tidak_sesuai_raw:
        if not isinstance(item, dict):
            continue
        canonical = normalize_program_name(str(item.get("nama_program") or ""))
        if canonical in allowed_programs and not rules.get(canonical, {}).get("eligible"):
            tidak_sesuai.append({
                "nama_program": canonical,
                "status": "TIDAK_ELIGIBLE",
                "alasan": rules.get(canonical, {}).get("reason") or item.get("alasan") or "Tidak memenuhi kriteria program.",
            })

    for item in rekomendasi_raw:
        if not isinstance(item, dict):
            continue
        current = item.copy()
        canonical = normalize_program_name(str(current.get("nama_program") or ""))
        
        if canonical not in allowed_programs:
            continue

        current["nama_program"] = canonical

        rule = rules.get(canonical)
        if rule and not rule["eligible"]:
            add_tidak_sesuai(canonical, rule["reason"])
            continue

        if rule and rule["eligible"]:
            current["status"] = "ELIGIBLE"
            current["sumber"] = current.get("sumber") or rule["source"]
            current["alasan_kelayakan"] = rule["reason"]
            add_rekomendasi(canonical, current["sumber"], current["alasan_kelayakan"])
            continue

        rekomendasi.append(current)

    # LLM kadang salah memasukkan program yang memenuhi hard rule ke
    # program_tidak_sesuai. Koreksi deterministik dilakukan di tahap akhir.
    for program_name, rule in rules.items():
        if rule["eligible"]:
            add_rekomendasi(program_name, rule["source"], rule["reason"])
        else:
            add_tidak_sesuai(program_name, rule["reason"])

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
    for key, program_name, source, sintesis, score, teknis in program_configs:
        kes = kesimpulan.get(key)
        inferred_layak = False
        if key == "pkh_plus":
            inferred_layak = (
                umur is not None and umur >= 70
                and desil is not None and desil <= 4
                and status_dtsen and "aktif" in str(status_dtsen).lower()
            )
            if inferred_layak:
                sintesis = (
                    f"Warga berusia {umur} tahun memenuhi syarat minimum lansia 70 tahun ke atas "
                    f"sesuai Juknis PKH Plus 2026 Pasal Sasaran Penerima. "
                    f"Desil nasional {desil} masuk klaster prioritas 1–4 dan status DTSEN tercatat aktif."
                )
            else:
                # Buat alasan deterministik berdasarkan kondisi yang gagal
                alasan_parts = []
                if umur is not None and umur < 70:
                    alasan_parts.append(
                        f"usia warga {umur} tahun belum memenuhi syarat minimum 70 tahun "
                        "yang ditetapkan Juknis PKH Plus 2026"
                    )
                elif umur is None:
                    alasan_parts.append("data usia warga tidak terdeteksi dari profil")
                if desil is not None and desil > 4:
                    alasan_parts.append(
                        f"desil nasional {desil} berada di luar klaster prioritas 1–4 PKH Plus"
                    )
                if not status_dtsen or "aktif" not in str(status_dtsen).lower():
                    alasan_parts.append("status DTSEN tidak aktif atau tidak terdeteksi")
                sintesis = (
                    "Tidak memenuhi kriteria PKH Plus: " + "; ".join(alasan_parts) + "."
                    if alasan_parts
                    else "Profil warga tidak memenuhi kriteria sasaran PKH Plus (Lanjut Usia 70+)."
                )
        elif key == "aspd":
            has_dis = profile_signals.get("has_disability")
            inferred_layak = (
                has_dis
                and umur is not None and umur <= 60
                and desil is not None and desil <= 5
            )
            if inferred_layak:
                sintesis = (
                    f"Warga memiliki indikasi hambatan fungsi/disabilitas yang tercatat pada profil, "
                    f"usia {umur} tahun masuk rentang sasaran ASPD (hingga 60 tahun), "
                    f"dan desil nasional {desil} masuk klaster prioritas sesuai Juklak ASPD 2026."
                )
            else:
                alasan_parts = []
                if not has_dis:
                    alasan_parts.append(
                        "tidak ditemukan indikasi hambatan fungsi/disabilitas pada profil warga "
                        "(semua dimensi fungsional: 'Tidak mengalami kesulitan')"
                    )
                if umur is not None and umur > 60:
                    alasan_parts.append(
                        f"usia warga {umur} tahun melebihi batas atas sasaran ASPD"
                    )
                if desil is not None and desil > 5:
                    alasan_parts.append(
                        f"desil nasional {desil} berada di luar klaster prioritas ASPD"
                    )
                sintesis = (
                    "Tidak memenuhi kriteria ASPD: " + "; ".join(alasan_parts) + "."
                    if alasan_parts
                    else "Profil warga tidak memenuhi kriteria sasaran ASPD berdasarkan data yang tersedia."
                )

        alasan = str(sintesis)
        if score is not None:
            alasan = f"{alasan} Skor Tim 1: {score}."

        if tim1_is_layak(kes) or inferred_layak:
            rekomendasi.append({
                "rank": rank,
                "nama_program": program_name,
                "status": "ELIGIBLE",
                "sumber": source_ref_for_program(results, source),
                "alasan_kelayakan": alasan,
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

    rekomendasi_names = [r["nama_program"] for r in rekomendasi]
    if rekomendasi_names:
        rekomendasi_teknis_narasi = (
            f"Rencana aksi operasional dan pendampingan di lapangan untuk program "
            f"{', '.join(rekomendasi_names)}. Penyaluran bantuan akan dikoordinasikan "
            f"oleh Dinas Sosial Provinsi Jawa Timur bersama pihak kelurahan/kecamatan setempat, "
            f"serta dilakukan monitoring dan evaluasi berkala untuk memastikan bantuan tepat sasaran."
        )
    else:
        rekomendasi_teknis_narasi = None

    return {
        "ringkasan_profil": ringkasan,
        "rekomendasi": rekomendasi,
        "rekomendasi_teknis_bansos": rekomendasi_teknis_narasi,
        "program_tidak_sesuai": tidak_sesuai,
    }
