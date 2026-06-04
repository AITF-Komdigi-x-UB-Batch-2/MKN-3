import os
import config
import requests
from dotenv import load_dotenv
from retrieval import PolicyRetriever

# inisialisasi constant
load_dotenv()
retriever = PolicyRetriever()

SYSTEM_PROMPT = {"role": "system", "content": "Anda adalah AI Auditor resmi Dinas Sosial Provinsi Jawa Timur yang bertugas melakukan verifikasi dan validasi kelayakan penerima manfaat dua program bantuan sosial: PKH Plus dan ASPD.\nBerdasarkan informasi kriteria yang diberikan, Anda wajib menghasilkan satu objek JSON murni (tanpa tag markdown) dengan struktur 'laporan_evaluasi' yang mencakup key 'profil_warga', 'kesimpulan', 'skor', 'analisis', dan 'parameter' secara konsisten."}

# no 8 di jsonl redit
QUERY = "<konteks>\nKRITERIA BANTUAN SOSIAL ASPD:\n1. Terdaftar penduduk Jatim (KK/KTP).\n2. Berusia 6 Bulan sd 60 Tahun.\n3. Penyandang disabilitas bed ridden / kesulitan aktifitas (ketergantungan fungsi hidup).\n4. Desil 1-5 prioritas (6-10 perlu verifikasi lapangan).\n5. Bukan klien panti/lembaga dan bukan penerima duplikasi APBD.\n\nKRITERIA BANTUAN SOSIAL PKH PLUS:\n1. Lansia 70 Tahun ke atas.\n2. Terdaftar dalam DTSEN desil 1-4.\n3. WNI ber-KTP/KK Provinsi Jawa Timur.\n4. Maksimal 1 lansia penerima per keluarga.\n</konteks>\n\nProfil Warga:\n- NIK / No. KK     : PRS_d3fbc8f3189a8bbc156e168d51fa1af6a33d4f8a35ad13618e1a201ca9c5bf97 / FAM_42a5ec4dde8712e33329cbf246147415b50fea445db3afd3b31a2454836dd011\n- Nama             : ****NEM\n- Umur             : 88 tahun\n- Hub. Kepala KK   : Kepala keluarga\n- Status Kawin     : Cerai hidup\n- Jml. Anggota KK  : 1 orang\n- Desil Nasional   : 1 | Status DTSEN: DTSEN AKTIF\n- Status Keberadaan: Ditemukan / Aktif\n- Bansos           : PKH, SEMBAKO\n- Kondisi Gizi     : Tidak diketahui\n- Penyakit Menahun : Tidak diketahui\n- Penglihatan      : Tidak mengalami kesulitan\n- Pendengaran      : Tidak mengalami kesulitan\n- Berjalan/Tangga  : Tidak mengalami kesulitan\n- Tangan/Jari      : Tidak mengalami kesulitan\n- Belajar/Intelek  : Tidak mengalami kesulitan\n- Perilaku         : Tidak mengalami kesulitan\n- Bicara/Komunikasi: Tidak mengalami kesulitan\n- Mengurus Diri    : Tidak mengalami kesulitan\n- Ingatan/Fokus    : Tidak mengalami kesulitan\n- Sedih/Depresi    : Tidak mengalami kesulitan\n- Wilayah          : Tamansatriyan, Kec. Tirtoyudo, Kabupaten Malang, Jawa Timur\n\nSkor Prioritas:\n- PKH+: 0.9212 | - ASPD: 0.0\nBuatkan laporan evaluasi kelayakannya secara utuh format JSON!"

MODEL_ENDPOINT = os.getenv('MODEL_ENDPOINT')
headers = {
    "Authorization": os.getenv('RUNPOD_API_KEY'),
    "Content-Type": "application/json"
}

# fase retrieval (context buat dimasukin ke system prompt)
retrieval = retriever.retrieve(query=QUERY, top_k=config.RETRIEVAL_TOP_K)

# fase augmentation
retrieval_text = "\n\n".join([doc.text for doc in retrieval])
augmented_system_prompt = (
    f"{SYSTEM_PROMPT['content']}\n\n"
    f"Berikut adalah konteks untuk Anda menentukan kebijakan:\n"
    f"<Konteks Policy>\n"
    f"{retrieval_text}\n"
    f"</Konteks Policy>"
)

# fase generation

## siapin payload
payload = {
    "model": os.getenv('MODEL_NAME'),
    "messages": [
        {"role": "system", "content": augmented_system_prompt},
        {"role": "user", "content": QUERY}
    ],
    "temperature": float(os.getenv('TEMPERATURE')),
    "max_tokens": int(os.getenv('MAX_TOKENS')),
    "stop": ["<lemmauser"]
}

## send request ke endpoint
response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers, timeout=300)
response.raise_for_status()

## print hasil
result_json = response.json()
result_text = result_json["choices"][0]["message"]["content"]
print(result_text)