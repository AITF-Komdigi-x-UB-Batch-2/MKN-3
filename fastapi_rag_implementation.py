import os
import config
from dotenv import load_dotenv
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval import PolicyRetriever

app = FastAPI(
    title="RAG MKN-3",
    version="0.3.0" # soalnya ws 3
)

retriever = PolicyRetriever()

SYSTEM_PROMPT_CONTENT = (
    "Anda adalah AI Auditor resmi Dinas Sosial Provinsi Jawa Timur yang bertugas "
    "melakukan verifikasi dan validasi kelayakan penerima manfaat dua program bantuan sosial: PKH Plus dan ASPD.\n"
    "Berdasarkan informasi kriteria yang diberikan, Anda wajib menghasilkan satu objek JSON murni "
    "(tanpa tag markdown) dengan struktur 'laporan_evaluasi' yang mencakup key "
    "'profil_warga', 'kesimpulan', 'skor', 'analisis', dan 'parameter' secara konsisten."
)
MODEL_ENDPOINT = os.getenv('MODEL_ENDPOINT')
headers = {
    "Authorization": os.getenv('RUNPOD_API_KEY'),
    "Content-Type": "application/json"
}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  # default 3 dokumen

@app.post("/query", response_model=dict)
async def process_query(request: QueryRequest):
    """
    Endpoint utama untuk query LLM dengan RAG.
    Input: JSON {'query': '...', 'top_k': ...}
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
            f"<Konteks Policy>\n"
            f"{retrieval_text}\n"
            f"</Konteks Policy>"
        )

        # 3. Generation
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

        # 4. Request ke LLM
        response = requests.post(
            MODEL_ENDPOINT,
            json=payload,
            headers=HEADERS,
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
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/retrieval')
async def retrieve_policy(request: QueryRequest):
    """pure retrieval tok"""
    try:
        retrieval = retriever.retrieve(query=request.query, top_k=request.top_k)
        retrieval_text = "\n\n".join([doc.text for doc in retrieval])
        return {
            "results": [
                {"text": doc.text, "score": doc.score, "metadata": doc.metadata} 
                for doc in retrieval
            ]
        }
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))