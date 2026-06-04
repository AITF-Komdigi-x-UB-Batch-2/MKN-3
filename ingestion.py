import time
import json
import config
import warnings
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv() # buat hf_token biar cepet

# ignore warnings karena ganggu
warnings.filterwarnings('ignore', category=UserWarning, module='qdrant_client')

COLLECTION_NAME = config.QDRANT_COLLECTION
MODEL_NAME = config.EMBED_MODEL_NAME

client = QdrantClient(url=config.QDRANT_URL)

client.set_model(MODEL_NAME)

# cek ada enggaknya collections
if not client.collection_exists(COLLECTION_NAME):
    print(f"Creating collection '{COLLECTION_NAME}' and starting ingestion...")
    
    # create
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=client.get_fastembed_vector_params()
    )

    # read
    documents = []
    metadata = []
    with open('chunked_data/juknis_extracted_normalized.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data_row = json.loads(line)
            documents.append(data_row['text'])
            metadata.append(data_row['metadata'])

    # pake payload indexing di kategori biar optimise filtering
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="nama_bansos",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

    # upsert (ingesting ke collection)
    start = time.time()
    VECTOR_NAME = list(client.get_fastembed_vector_params().keys())[0]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector={
                    VECTOR_NAME: models.Document(
                        text=doc, model=MODEL_NAME
                    )
                },
                payload={**meta, "text": doc}
            )
            for i, (doc, meta) in enumerate(zip(documents, metadata))
        ]
    )
    end = time.time()
    print(f"Ingestion completed successfully in {end - start:.2f} seconds!")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping ingestion.")