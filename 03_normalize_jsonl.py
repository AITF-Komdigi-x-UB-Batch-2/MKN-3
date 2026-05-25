# !!! DEPRECATED !!!
# File ini sudah tidak digunakan.

import os
import json
import re
from pathlib import Path
from config import CHUNKED_DIR

# Kamus singkatan yang sering muncul di dataset
ABBREVIATIONS = {
    "PKH": "Program Keluarga Harapan",
    "BPNT": "Bantuan Pangan Non Tunai",
    "KPM": "Keluarga Penerima Manfaat",
    "BLT": "Bantuan Langsung Tunai",
    "KKS": "Kartu Keluarga Sejahtera",
    "KPA": "Kuasa Pengguna Anggaran",
    "PPKE": "Percepatan Penghapusan Kemiskinan Ekstrem",
    "BBM": "Bahan Bakar Minyak",
    "APBN": "Anggaran Pendapatan dan Belanja Negara",
    "APBD": "Anggaran Pendapatan dan Belanja Daerah",
    "DTKS": "Data Terpadu Kesejahteraan Sosial",
    "SPM": "Surat Perintah Membayar",
    "PAUD": "Pendidikan Anak Usia Dini",
    "TKPK": "Tim Koordinasi Penanggulangan Kemiskinan",
    "KPPN": "Kantor Pelayanan Perbendaharaan Negara",
    "KUR": "Kredit Usaha Rakyat",
    "BUMN": "Badan Usaha Milik Negara",
    "RPL": "Rekening Pemerintah Lainnya",
    "TNI": "Tentara Nasional Indonesia",
    "APIP": "Aparat Pengawasan Intern Pemerintah",
    "JKN": "Jaminan Kesehatan Nasional",
    "PUPR": "Pekerjaan Umum dan Perumahan Rakyat",
    "PIP": "Program Indonesia Pintar",
    "KIP": "Kartu Indonesia Pintar",
    "ASN": "Aparatur Sipil Negara",
    "SIKS-NG": "Sistem Informasi Kesejahteraan Sosial Next-Generation",
    "OM-SPAN": "Online Monitoring Sistem Perbendaharaan dan Anggaran Negara"
}

# Compile regex untuk pencarian kata yang utuh (word boundary \b)
ABBR_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in ABBREVIATIONS.keys()) + r')\b')

def expand_abbreviations(text):
    """
    Mengganti singkatan di dalam teks menjadi format: Kepanjangan (Singkatan).
    Contoh: 'KPM' -> 'Keluarga Penerima Manfaat (KPM)'
    """
    if not isinstance(text, str):
        return text
    return ABBR_PATTERN.sub(lambda match: f"{ABBREVIATIONS[match.group(0)]} ({match.group(0)})", text)

def normalize_text(text):
    """
    Membersihkan teks dari spasi ganda, newline (\n), carriage return (\r), 
    atau karakter whitespace tak terlihat agar menjadi teks yang rapi.
    """
    if not isinstance(text, str):
        return text
    # Mengganti karakter whitespace (\n, \r, \t, multiple spaces) menjadi spasi tunggal
    text = re.sub(r'\s+', ' ', text)
    # Menghapus whitespace di awal dan akhir kalimat
    return text.strip()

def normalize_jsonl_file(filepath):
    # Lewati file yang sudah dinormalisasi agar tidak terproses ganda
    if filepath.name.endswith('_normalized.jsonl'):
        return
        
    print(f"Memproses file: {filepath.name}")
    
    normalized_lines = []
    
    # 1. Membaca dan melakukan normalisasi dari file
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Normalisasi pada key utama 'text'
                if 'text' in data:
                    text = expand_abbreviations(data['text'])
                    data['text'] = normalize_text(text)
                
                # Normalisasi pada setiap value di dalam key 'metadata'
                if 'metadata' in data and isinstance(data['metadata'], dict):
                    for key, value in data['metadata'].items():
                        if isinstance(value, str):
                            val = expand_abbreviations(value)
                            data['metadata'][key] = normalize_text(val)
                
                normalized_lines.append(data)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing JSON di {filepath.name} baris {line_num}: {e}")
                
    # 2. Menentukan nama file output baru
    output_path = filepath.with_name(f"{filepath.stem}_normalized.jsonl")
        
    # 3. Menulis file output dengan data yang sudah bersih
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in normalized_lines:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ Selesai normalisasi: {filepath.name} -> {output_path.name}")
    except Exception as e:
        print(f"❌ Gagal menulis file {output_path.name}. Error: {e}")

def main():
    chunked_dir = Path(CHUNKED_DIR)
    print(f"📂 Folder target: {chunked_dir}")
    
    if not chunked_dir.exists() or not chunked_dir.is_dir():
        print(f"❌ Folder {chunked_dir} tidak ditemukan.")
        return
        
    # Mencari seluruh file berakhiran .jsonl
    jsonl_files = list(chunked_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"ℹ️ Tidak ada file .jsonl yang ditemukan di {chunked_dir}")
        return
        
    print(f"🔍 Menemukan {len(jsonl_files)} file .jsonl untuk dinormalisasi.\n")
    
    for filepath in jsonl_files:
        normalize_jsonl_file(filepath)
        
    print("\n🎉 Proses normalisasi berhasil diselesaikan di seluruh file!")

if __name__ == "__main__":
    main()
