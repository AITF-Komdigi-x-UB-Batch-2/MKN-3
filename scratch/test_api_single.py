import httpx
import json

url = "http://localhost:8002/retrieve"
payload = {
    "content": "Profil Warga:\n- NIK / No. KK     : PRS_4d633b26e89711bea6b54b466829d869ee53f0ec779b932dc6c94f4d0cee1aa9 / FAM_2aa6d00cca29d81b166b4268eb52cac87cfddbfaad71588bf29433e55f0624a6\n- Nama             : ***********A K\n- Umur             : 50 tahun\n- Hub. Kepala KK   : Kepala keluarga\n- Status Kawin     : Kawin\n- Jml. Anggota KK  : 2 orang\n- Desil Nasional   : 10 | Status DTSEN: DTSEN AKTIF\n- Status Keberadaan: Ditemukan / Aktif\n- Bansos           : -\n- Kondisi Gizi     : Tidak diketahui\n- Penyakit Menahun : Tidak diketahui\n- Penglihatan      : Tidak mengalami kesulitan\n- Pendengaran      : Tidak mengalami kesulitan\n- Berjalan/Tangga  : Tidak mengalami kesulitan\n- Tangan/Jari      : Tidak mengalami kesulitan\n- Belajar/Intelek  : Tidak mengalami kesulitan\n- Perilaku         : Tidak mengalami kesulitan\n- Bicara/Komunikasi: Tidak mengalami kesulitan\n- Mengurus Diri    : Tidak mengalami kesulitan\n- Ingatan/Fokus    : Tidak mengalami kesulitan\n- Sedih/Depresi    : Tidak mengalami kesulitan\n- Wilayah          : Karangwidoro, Kec. Dau, Kabupaten Malang, Jawa Timur\n\nBuatkan laporan evaluasi kelayakannya secara utuh format JSON!",
    "filter_programs_only": True
}

try:
    response = httpx.post(url, json=payload, timeout=10.0)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
