import os
import logging
from dotenv import load_dotenv
from operator import itemgetter

from langchain_nvidia_ai_endpoints import  NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from uuid import uuid4
from pymilvus import MilvusClient, utility


from .process_files import read_file
from .utils import (
    get_vectorstore,
    get_embedder,
)
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# Configuration
DEFAULT_MILVUS_HOST = "127.0.0.1"
DEFAULT_MILVUS_PORT = "19530"
collection_name = "default_collection"
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"
NLIST = 128
SEARCH_NPROBE = 10

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

SEARCH_LIMIT = 50
TOP_K = 5
EMBEDDING_MODEL_NVIDIA = "nvidia/nv-embedqa-e5-v5"
vs = None

document_embedder =  get_embedder(EMBEDDING_MODEL_NVIDIA)
logger.info(f"Document Embedder: {document_embedder}.")
logger.info("Document embedder initialized successfully.")


def indexing(filename: str, filepath: str = "../Data"):
    try:
        # client = MilvusClient(
        #     uri="http://localhost:19530",
        #     token="root:Milvus"
        # )
        absolute_path = os.path.abspath(os.path.join(filepath))
        
        global vs
        vs = get_vectorstore(vs, document_embedder=document_embedder)
        logger.info("Vector Database initialized.")

        data = read_file(absolute_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,
            chunk_overlap=500
        )
        chunks = text_splitter.split_text(data)
        
        def validate_and_split(chunk):
            if len(chunk) > 65535:
                logger.warning(f"Chunk size {len(chunk)} exceeds VARCHAR limit. Splitting at character level.")
                return [chunk[i:i+65000] for i in range(0, len(chunk), 65000)]
            return [chunk]

        final_chunks = []
        for chunk in chunks:
            final_chunks.extend(validate_and_split(chunk))
        
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": filename, "chunk_index": idx}
            )
            for idx, chunk in enumerate(final_chunks)
        ]
    
        uuids = [str(uuid4()) for _ in range(len(documents))]
        logger.info(f"Generated {len(uuids)} UUIDs for {len(documents)} documents.")
        connections.connect(alias="default", host="localhost", port="19530")
        collection = Collection(name="default_collection")

        batch_size = 50
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = uuids[i:i+batch_size]
            try:
                logger.info(f"Adding batch {i // batch_size + 1}...")
                
                batch = [
                    [doc.metadata['source'] for doc in batch_docs],
                    [doc.metadata['chunk_index'] for doc in batch_docs],
                    [doc.page_content for doc in batch_docs],
                    [[0.0] * 1024 for _ in batch_docs]
                ]
                
                collection.insert(batch)
                logger.info(f"Added batch {i // batch_size +1}.")
                entity_count = collection.num_entities
                logger.info(f"Total number of vectors in the collection 'Default Collection': {entity_count}.")
            except Exception as e:
                logger.error(f"Failed to add batch {i // batch_size + 1}: {e}")
                raise
        # Flush to ensure data is persisted
        collection.flush()
        logger.info("Data flushed to disk.")

        # Load the collection into memory for querying
        collection.load()
        logger.info(f"Collection '{collection_name}' loaded into memory.")

        # Log the number of entities in the collection
        entity_count = collection.num_entities
        logger.info(f"Total number of vectors in the collection '{collection_name}': {entity_count}.")

        logger.info("Indexing completed successfully.")
        return "Indexing Successful"
    
    except Exception as e:
        vs = None
        logger.error(f"indexing Failed: {e}")
        raise e


# def indexing(filename: str, filepath: str = "../Data"):
#     try:
#         absolute_path = os.path.abspath(os.path.join(filepath))
        
#         global vs
#         vs = get_vectorstore(vs, document_embedder=document_embedder)
#         logger.info("Vector Database initialized.")

#         data = read_file(absolute_path)

#         # Initialize text splitter
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=40000,
#             chunk_overlap=500
#         )
#         chunks = text_splitter.split_text(data)

#         # Validate and split chunks if exceeding VARCHAR limits
#         def validate_and_split(chunk):
#             if len(chunk) > 65535:
#                 logger.warning(f"Chunk size {len(chunk)} exceeds VARCHAR limit. Splitting at character level.")
#                 return [chunk[i:i + 65000] for i in range(0, len(chunk), 65000)]
#             return [chunk]

#         # Process chunks
#         final_chunks = []
#         for chunk in chunks:
#             final_chunks.extend(validate_and_split(chunk))

#         # Create documents with metadata
#         documents = [
#             {
#                 "id": str(uuid4()),
#                 "source": filename,
#                 "page_content": chunk,
#                 "vector": [0.0] * 1024  # Replace with actual embeddings if available
#             }
#             for idx, chunk in enumerate(final_chunks)
#         ]

#         logger.info(f"Generated {len(documents)} documents for indexing.")

#         # Connect to Milvus
#         connections.connect(alias="default", host="localhost", port="19530")
#         collection_name = "default_collection"
#         collection = Collection(name=collection_name)

#         # Batch insert data
#         batch_size = 50
#         for i in range(0, len(documents), batch_size):
#             batch_docs = documents[i:i + batch_size]
#             try:
#                 logger.info(f"Inserting batch {i // batch_size + 1}...")

#                 # Prepare batch data for insertion
#                 ids = [doc["id"] for doc in batch_docs]
#                 sources = [doc["source"] for doc in batch_docs]
#                 page_contents = [doc["page_content"] for doc in batch_docs]
#                 vectors = [doc["vector"] for doc in batch_docs]

#                 # Insert into collection
#                 collection.insert([ids, sources, page_contents, vectors])
#                 logger.info(f"Batch {i // batch_size + 1} inserted successfully.")
#             except Exception as e:
#                 logger.error(f"Failed to insert batch {i // batch_size + 1}: {e}")
#                 raise

#         # Flush to ensure data is persisted
#         collection.flush()
#         logger.info("Data flushed to disk.")

#         # Create index on the vector field for efficient querying
#         collection.create_index(
#             field_name="vector",
#             index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
#         )
#         logger.info("Index created successfully.")

#         # Load the collection into memory
#         collection.load()
#         logger.info(f"Collection '{collection_name}' loaded into memory.")

#         # Log the total number of entities
#         entity_count = collection.num_entities
#         logger.info(f"Total number of vectors in the collection '{collection_name}': {entity_count}.")

#         logger.info("Indexing completed successfully.")
#         return "Indexing Successful"

#     except Exception as e:
#         logger.error(f"Indexing failed: {e}")
#         raise

def openai_ingestion(filename:str, filepath:str="../Data"):
    try:
        absolute_path = os.path.abspath(filename)
        print("Absolute path in get_documents:", absolute_path)
        print("Does file exist:", os.path.exists(absolute_path))
        data = read_file(absolute_path)

        chunks = text_splitter.split_text(data)

        global vs
        vs = get_vectorstore(vs, document_embedder)
        
        documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
        vs.add_documents(documents)
        logger.info("Ingestion completed successfully.")
        return "Ingestion Successful"
    except Exception as e:
        vs = None
        logger.info(f"Ingestion Failed: {e}")

