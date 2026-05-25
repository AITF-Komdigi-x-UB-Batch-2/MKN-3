import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_COLLECTION, configure_utf8_stdio

configure_utf8_stdio()

def main():
    print(f"Menyambungkan ke Qdrant di {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    
    existing = [c.name for c in client.get_collections().collections]
    
    if QDRANT_COLLECTION in existing:
        print(f"⚠️ Collection '{QDRANT_COLLECTION}' ditemukan.")
        print("Menghapus semua vector lama...")
        
        # Hapus collection dan semua datanya
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        print("✅ Collection berhasil dihapus! Ruang Anda sekarang kosong.")
        print("\nAnda kini bisa mempopulasi ulang DB dengan lancar via:")
        print("python 04_embed_and_ingest_v2.py")
    else:
        print(f"Collection '{QDRANT_COLLECTION}' tidak ditemukan, DB memang sudah kosong.")

if __name__ == "__main__":
    main()
