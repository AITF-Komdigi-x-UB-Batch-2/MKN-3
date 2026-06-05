import os
import config
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval import PolicyRetriever

load_dotenv()

app = FastAPI(
    title="RAG MKN-3",
    version="0.3.0" # soalnya ws 3
)

retriever = PolicyRetriever()

SYSTEM_PROMPT_CONTENT = os.getenv('SYSTEM_PROMPT')
MODEL_ENDPOINT = os.getenv('MODEL_ENDPOINT')
headers = {
    "Authorization": os.getenv('RUNPOD_API_KEY'),
    "Content-Type": "application/json"
}

class QueryRequest(BaseModel):
    query: str
    top_k: int = config.RETRIEVAL_TOP_K
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": ("<konteks>\nKRITERIA BANTUAN SOSIAL ASPD:\n1. Terdaftar penduduk Jatim (KK/KTP).\n2. Berusia 6 Bulan sd 60 Tahun.\n3. Penyandang disabilitas bed ridden / kesulitan aktifitas (ketergantungan fungsi hidup).\n4. Desil 1-5 prioritas (6-10 perlu verifikasi lapangan).\n5. Bukan klien panti/lembaga dan bukan penerima duplikasi APBD.\n\nKRITERIA BANTUAN SOSIAL PKH PLUS:\n1. Lansia 70 Tahun ke atas.\n2. Terdaftar dalam DTSEN desil 1-4.\n3. WNI ber-KTP/KK Provinsi Jawa Timur.\n4. Maksimal 1 lansia penerima per keluarga.\n</konteks>\n\nProfil Warga:\n- NIK / No. KK     : PRS_d3fbc8f3189a8bbc156e168d51fa1af6a33d4f8a35ad13618e1a201ca9c5bf97 / FAM_42a5ec4dde8712e33329cbf246147415b50fea445db3afd3b31a2454836dd011\n- Nama             : ****NEM\n- Umur             : 88 tahun\n- Hub. Kepala KK   : Kepala keluarga\n- Status Kawin     : Cerai hidup\n- Jml. Anggota KK  : 1 orang\n- Desil Nasional   : 1 | Status DTSEN: DTSEN AKTIF\n- Status Keberadaan: Ditemukan / Aktif\n- Bansos           : PKH, SEMBAKO\n- Kondisi Gizi     : Tidak diketahui\n- Penyakit Menahun : Tidak diketahui\n- Penglihatan      : Tidak mengalami kesulitan\n- Pendengaran      : Tidak mengalami kesulitan\n- Berjalan/Tangga  : Tidak mengalami kesulitan\n- Tangan/Jari      : Tidak mengalami kesulitan\n- Belajar/Intelek  : Tidak mengalami kesulitan\n- Perilaku         : Tidak mengalami kesulitan\n- Bicara/Komunikasi: Tidak mengalami kesulitan\n- Mengurus Diri    : Tidak mengalami kesulitan\n- Ingatan/Fokus    : Tidak mengalami kesulitan\n- Sedih/Depresi    : Tidak mengalami kesulitan\n- Wilayah          : Tamansatriyan, Kec. Tirtoyudo, Kabupaten Malang, Jawa Timur\n\nSkor Prioritas:\n- PKH+: 0.9212 | - ASPD: 0.0\nBuatkan laporan evaluasi kelayakannya secara utuh format JSON!"),
                    "top_k": 5,
                }
            ]
        }
    }
@app.post("/recommend", response_model=dict)
async def process_query(request: QueryRequest):
    """
    Endpoint utama untuk query LLM dengan RAG.
    Input: JSON {'query': '...', 'top_k': 5}
    Output: JSON hasil LLM
    """
    try:
        # 1. Retrieval
        retrieval = retriever.retrieve(query=request.query, top_k=request.top_k)
        retrieval_text = "\n\n".join([doc.text for doc in retrieval])

        # 2. Augmentation
        augmented_system_prompt = (
            f"{SYSTEM_PROMPT_CONTENT}\n\n"
            f"Berikut adalah konteks untuk Anda menentukan kebijakan:\n"
            f"<Konteks>\n"
            f"{retrieval_text}\n"
            f"</Konteks>"
        )

        # 3. Generation
        payload = {
            "model": os.getenv('MODEL_NAME'),
            "messages": [
                {"role": "system", "content": augmented_system_prompt},
                {"role": "user", "content": request.query}
            ],
            "temperature": os.getenv('TEMPERATURE'),
            "max_tokens": os.getenv('MAX_TOKENS'),
            "stop": ["<lemmauser"]
        }

        # 4. Request ke LLM
        response = requests.post(
            str(MODEL_ENDPOINT),
            json=payload,
            headers=headers,
            timeout=300
        )
        response.raise_for_status()

        result_json = response.json()
        result_text = result_json["choices"][0]["message"]["content"]

        # Optional: Parse JSON result biar lebih rapi
        try:
            # clean markdown code block
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
        except Exception as e:
            print(f"Warning: gagal clean markdown, pakai mentah: {e}")

        return {"response": result_text}

    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e)
        ) from e

@app.post('/retrieval')
async def retrieve_policy(request: QueryRequest):
    """pure retrieval tok"""
    try:
        retrieval = retriever.retrieve(query=request.query, top_k=request.top_k)
        return {
            "results": [
                {"text": doc.text, "score": doc.score, "metadata": doc.metadata} 
                for doc in retrieval
            ]
        }
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=e.response.status_code, 
            detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        'fastapi_rag_implementation:app',
        host='0.0.0.0',
        port=8000,
        reload=True
    )