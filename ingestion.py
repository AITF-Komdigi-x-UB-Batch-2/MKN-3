import time
import json
import config
import logging
import warnings
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv() # buat hf_token biar cepet

# ignore warnings karena ganggu
warnings.filterwarnings('ignore', category=UserWarning, module='qdrant_client')

# config
COLLECTION_NAME = config.QDRANT_COLLECTION
MODEL_NAME = config.EMBED_MODEL_NAME
client = QdrantClient(url=config.QDRANT_URL)
client.set_model(MODEL_NAME)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def collections_non_null(collection_name: str):
    if client.collection_exists(collection_name):
        if client.get_collection(collection_name).points_count == 0:
            return False
        else:
            return True
    else:
        return False

def ingest(collection_name: str):
    if collections_non_null(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Skipping ingestion.")
    else:
        logger.info(f"Collection '{collection_name}' does not exist.")
        # delete collection 
        client.delete_collection(collection_name)
    
        # create
        client.create_collection(
            collection_name=collection_name,
            vectors_config=client.get_fastembed_vector_params()
        )

        # read
        documents = []
        metadata = []
        with open('chunked_data/juknis_extracted_normalized.jsonl', 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='Reading JSONL'):
                if not line.strip():
                    continue
                data_row = json.loads(line)
                meta = data_row.get('metadata', {})
                if meta.get('retrieval_priority') == 'low':
                    continue
                documents.append(data_row['text'])
                metadata.append(meta)

        # pake payload indexing di kategori biar optimise filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="nama_bansos",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        start = time.time()
        VECTOR_NAME = list(client.get_fastembed_vector_params().keys())[0]
        points = []

        for i, (doc, meta) in enumerate(
            tqdm(zip(documents, metadata),
                total=len(documents),
                desc="Building points")
        ):
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        VECTOR_NAME: models.Document(
                            text=doc,
                            model=MODEL_NAME
                        )
                    },
                    payload={**meta, "text": doc}
                )
            )

        # upsert (ingesting ke collection)
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        end = time.time()
        print(f"Ingestion completed successfully in {end - start:.2f} seconds!")

# entrypoint
if __name__ == "__main__":
    ingest(COLLECTION_NAME)