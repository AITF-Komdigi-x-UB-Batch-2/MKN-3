from qdrant_client import QdrantClient, models
import json
import time
import warnings
import config

warnings.filterwarnings('ignore')
client = QdrantClient(url=config.QDRANT_URL)

collection_name = config.QDRANT_COLLECTION
model_name = config.EMBED_MODEL_NAME

client.set_model(model_name)
vector_name = list(client.get_fastembed_vector_params().keys())[0]

prompt_sample_pkh = {"messages": [{"role": "system", "content": "Anda adalah AI Auditor resmi Dinas Sosial Provinsi Jawa Timur yang bertugas melakukan verifikasi dan validasi kelayakan penerima manfaat dua program bantuan sosial: PKH Plus dan ASPD.\nBerdasarkan informasi kriteria yang diberikan, Anda wajib menghasilkan satu objek JSON murni (tanpa tag markdown) dengan struktur 'laporan_evaluasi' yang mencakup key 'profil_warga', 'kesimpulan', 'skor', 'analisis', dan 'parameter' secara konsisten."}, {"role": "user", "content": "<konteks>\nKRITERIA BANTUAN SOSIAL ASPD:\n1. Terdaftar penduduk Jatim (KK/KTP).\n2. Berusia 6 Bulan sd 60 Tahun.\n3. Penyandang disabilitas bed ridden / kesulitan aktifitas (ketergantungan fungsi hidup).\n4. Desil 1-5 prioritas (6-10 perlu verifikasi lapangan).\n5. Bukan klien panti/lembaga dan bukan penerima duplikasi APBD.\n\nKRITERIA BANTUAN SOSIAL PKH PLUS:\n1. Lansia 70 Tahun ke atas.\n2. Terdaftar dalam DTSEN desil 1-4.\n3. WNI ber-KTP/KK Provinsi Jawa Timur.\n4. Maksimal 1 lansia penerima per keluarga.\n</konteks>\n\nProfil Warga:\n- NIK / No. KK     : PRS_d3fbc8f3189a8bbc156e168d51fa1af6a33d4f8a35ad13618e1a201ca9c5bf97 / FAM_42a5ec4dde8712e33329cbf246147415b50fea445db3afd3b31a2454836dd011\n- Nama             : ****NEM\n- Umur             : 88 tahun\n- Hub. Kepala KK   : Kepala keluarga\n- Status Kawin     : Cerai hidup\n- Jml. Anggota KK  : 1 orang\n- Desil Nasional   : 1 | Status DTSEN: DTSEN AKTIF\n- Status Keberadaan: Ditemukan / Aktif\n- Bansos           : PKH, SEMBAKO\n- Kondisi Gizi     : Tidak diketahui\n- Penyakit Menahun : Tidak diketahui\n- Penglihatan      : Tidak mengalami kesulitan\n- Pendengaran      : Tidak mengalami kesulitan\n- Berjalan/Tangga  : Tidak mengalami kesulitan\n- Tangan/Jari      : Tidak mengalami kesulitan\n- Belajar/Intelek  : Tidak mengalami kesulitan\n- Perilaku         : Tidak mengalami kesulitan\n- Bicara/Komunikasi: Tidak mengalami kesulitan\n- Mengurus Diri    : Tidak mengalami kesulitan\n- Ingatan/Fokus    : Tidak mengalami kesulitan\n- Sedih/Depresi    : Tidak mengalami kesulitan\n- Wilayah          : Tamansatriyan, Kec. Tirtoyudo, Kabupaten Malang, Jawa Timur\n\nSkor Prioritas:\n- PKH+: 0.9212 | - ASPD: 0.0\nBuatkan laporan evaluasi kelayakannya secara utuh format JSON!"}, {"role": "assistant", "content": "{\"laporan_evaluasi\": {\"profil_warga\": {\"nik\": \"PRS_d3fbc8f3189a8bbc156e168d51fa1af6a33d4f8a35ad13618e1a201ca9c5bf97\", \"no_kk\": \"FAM_42a5ec4dde8712e33329cbf246147415b50fea445db3afd3b31a2454836dd011\", \"nama\": \"****NEM\", \"umur\": 88, \"hubungan_kepala_keluarga\": \"Kepala keluarga\", \"status_perkawinan\": \"Cerai hidup\", \"jumlah_anggota_keluarga\": 1, \"status_dtsen\": \"DTSEN AKTIF\", \"wilayah\": {\"provinsi\": \"Jawa Timur\", \"kabupaten_kota\": \"Kabupaten Malang\", \"kecamatan\": \"Tirtoyudo\", \"kelurahan_desa\": \"Tamansatriyan\"}}, \"kesimpulan\": {\"pkh_plus\": {\"status_kelayakan\": \"LAYAK\", \"label\": 1}, \"aspd\": {\"status_kelayakan\": \"TIDAK LAYAK\", \"label\": 0}}, \"skor\": {\"skor_pkh_plus\": 0.9212, \"skor_aspd\": 0.0}, \"analisis\": {\"demografi\": \"Warga berusia 88 tahun masuk kategori lansia dengan kerentanan sosial-kesehatan tinggi. Posisi 'Kepala keluarga' merupakan pilar utama rumah tangga; kerentanan pada posisi ini berdampak langsung dan merusak ketahanan ekonomi seluruh anggota keluarga.\", \"ekonomi\": \"Desil 1 menempatkan warga di kategori miskin ekstrem (prioritas utama).\", \"kesehatan_gizi\": \"Tidak ditemukan indikator kesehatan khusus.\", \"disabilitas_fungsi\": \"Tidak ditemukan hambatan fungsi signifikan.\", \"sintesis_pkh_plus\": \"Warga MEMENUHI seluruh kriteria PKH Plus Jawa Timur. Faktor yang dipenuhi: usia lansia terpenuhi (88 tahun); DTSEN pada desil prioritas (1); memiliki NIK dan KK Provinsi Jawa Timur yang terdata; merupakan lansia yang berhak dalam 1 nomor KK. Intervensi PKH Plus tepat sasaran.\", \"sintesis_aspd\": \"Warga TIDAK MEMENUHI kriteria ASPD. Alasan gugur: usia 88 tahun di luar rentang persyaratan; tidak ditemukan hambatan fungsional berat indikator ketergantungan/bed-ridden. \"}, \"parameter\": {\"desil_nasional\": 1, \"status_keberadaan\": \"Ditemukan / Aktif\", \"riwayat_bansos\": \"PKH, SEMBAKO\", \"kesehatan\": {\"kondisi_gizi\": \"Tidak diketahui\", \"penyakit_menahun\": \"Tidak diketahui\"}, \"disabilitas\": {\"penglihatan\": \"Tidak mengalami kesulitan\", \"pendengaran\": \"Tidak mengalami kesulitan\", \"berjalan_naik_tangga\": \"Tidak mengalami kesulitan\", \"menggunakan_tangan_jari\": \"Tidak mengalami kesulitan\", \"belajar_intelektual\": \"Tidak mengalami kesulitan\", \"pengendalian_perilaku\": \"Tidak mengalami kesulitan\", \"berbicara_komunikasi\": \"Tidak mengalami kesulitan\", \"mengurus_diri\": \"Tidak mengalami kesulitan\", \"mengingat_berkonsentrasi\": \"Tidak mengalami kesulitan\", \"kesedihan_depresi\": \"Tidak mengalami kesulitan\"}}}}"}]}

# karena yang dipake nanti cuma metadata 'content' dari 'user'
QUERY = prompt_sample_pkh['messages'][1]['content']

start_time = time.time()
response = client.query_points(
    collection_name,
    query=models.Document(text=QUERY, model=model_name),
    using=vector_name,
    limit=5
).points
end_time = time.time() - start_time

# save hasil retrieval ke jsonl
output_file = 'pkh_k5.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for h in response:
        f.write(json.dumps(h.payload) + '\n')

print(f"Berhasil menyimpan {len(response)} dokumen ke '{output_file}'!")

for h in response:
    print(json.dumps(h.payload, indent=2))

print(f'waktu retrieval: {end_time:.2f} detik')