import config
from qdrant_client import QdrantClient

client = QdrantClient(url=config.QDRANT_URL)
client.collection_exists(config.QDRANT_COLLECTION)
client.delete_collection(config.QDRANT_COLLECTION)

print(f'collection {config.QDRANT_COLLECTION} dihapus')