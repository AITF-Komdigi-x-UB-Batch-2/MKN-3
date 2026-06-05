import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
import httpx
from tqdm.asyncio import tqdm_asyncio

# Target API endpoint
DEFAULT_API_URL = "http://76.13.194.250:8000/retrieve"
DEFAULT_INPUT = "eval_queries.jsonl" # ini harus diganti buat beda file 
DEFAULT_OUTPUT = "retrieval_results_eval.jsonl" # ini juga

def extract_nik(query_profil: str) -> str:
    """Extract NIK from the query profile text."""
    # Matches NIK or NIK / No. KK followed by spaces/colons and the value
    match = re.search(r"NIK\s*(?:/\s*No\.\s*KK)?\s*:\s*([^\s/]+)", query_profil, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

async def retrieve_query(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, url: str, query_profil: str, index: int, filter_programs: bool, top_k: int, top_n: int):
    nik = extract_nik(query_profil)
    
    # Payload for the /retrieve endpoint
    payload = {
        "content": query_profil,
        "filter_programs_only": filter_programs
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if top_n is not None:
        payload["top_n"] = top_n

    async with semaphore:
        for attempt in range(3):  # Up to 3 retries
            try:
                response = await client.post(url, json=payload, timeout=60.0)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Format matching batch_retrieval.py for compatibility
                    retrieved_chunks = []
                    for r in data.get("results", []):
                        meta = r.get("metadata", {})
                        meta["sumber"] = r.get("sumber", "unknown")
                        meta["judul_halaman"] = r.get("judul_halaman")
                        meta["page_number"] = r.get("page_number")
                        
                        retrieved_chunks.append({
                            "text": r.get("text", ""),
                            "score": r.get("rerank_score", 0.0),
                            "embed_score": r.get("embed_score", 0.0),
                            "metadata": meta
                        })
                    
                    return {
                        "nik": nik,
                        "query_profil": query_profil,
                        "num_chunks": len(retrieved_chunks),
                        "retrieved_chunks": retrieved_chunks,
                        "success": True
                    }
                else:
                    await asyncio.sleep(1.0 * (attempt + 1))
            except Exception:
                await asyncio.sleep(1.0 * (attempt + 1))
        
        # Fallback if failed
        return {
            "nik": nik,
            "query_profil": query_profil,
            "num_chunks": 0,
            "retrieved_chunks": [],
            "success": False
        }

async def process_batch(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)
        
    queries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                query_profil = obj.get("query_profil")
                if query_profil:
                    queries.append(query_profil)
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON line {line_num} skipped: {e}")
                
    total_queries = len(queries)
    print(f"📂 Loaded {total_queries} queries from {input_path}")
    print(f"🔗 Target Endpoint: {args.url}")
    print(f"🔒 Filter Programs: {args.filter_programs}")
    print(f"📈 Concurrency Limit: {args.concurrency}")
    
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency * 2)
    
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            retrieve_query(client, semaphore, args.url, q, i, args.filter_programs, args.top_k, args.top_n)
            for i, q in enumerate(queries)
        ]
        
        # Run concurrently and show progress bar
        results = await tqdm_asyncio.gather(*tasks)
        
    # Write output to JSONL
    success_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for r in results:
            if r["success"]:
                success_count += 1
            # Pop success flag so it's not written to final jsonl
            r.pop("success")
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Processing complete!")
    print(f"   Total queries : {total_queries}")
    print(f"   Success count : {success_count}")
    print(f"   Failed count  : {total_queries - success_count}")
    print(f"   Output saved to: {output_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Batch query the SIRA retrieval endpoint for eval_queries.jsonl")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input JSONL file (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output JSONL file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--url", default=DEFAULT_API_URL, help=f"Retrieval API endpoint URL (default: {DEFAULT_API_URL})")
    parser.add_argument("--concurrency", type=int, default=20, help="Max number of concurrent requests (default: 20)")
    parser.add_argument("--no-filter", action="store_false", dest="filter_programs", help="Disable program filtering")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-K parameter")
    parser.add_argument("--top-n", type=int, default=None, help="Override top-N parameter")
    
    args = parser.parse_args()
    
    t_start = time.time()
    asyncio.run(process_batch(args))
    print(f"⏱️ Total time elapsed: {time.time() - t_start:.2f} seconds")

if __name__ == "__main__":
    main()
