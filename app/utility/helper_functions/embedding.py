import os
import logging
from dotenv import load_dotenv
from operator import itemgetter

from langchain_nvidia_ai_endpoints import  NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from uuid import uuid4
from pymilvus import Collection, connections
from pymilvus import MilvusClient


from .process_files import read_file
from .utils import (
    create_vectorstore_langchain,
    get_vectorstore,
    get_embedder,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

SEARCH_LIMIT = 50  # Adjusted based on reference code
TOP_K = 5  # Number of top documents to use
EMBEDDING_MODEL_NVIDIA = "nvidia/nv-embedqa-e5-v5"
vs = None

# Initialize Embedding Object
document_embedder =  get_embedder(EMBEDDING_MODEL_NVIDIA)
logger.info(f"Document Embedder: {document_embedder}.")
logger.info("Document embedder initialized successfully.")


def ingestion(filename: str, filepath: str = "../Data"):
    try:
        client = MilvusClient(
            uri="http://localhost:19530",
            token="root:Milvus"
        )
        absolute_path = os.path.abspath(os.path.join(filepath))
        
        global vs
        vs = get_vectorstore(vs, document_embedder=document_embedder)
        logger.info("Vector Database initialized.")

        # Read file and chunk text to respect VARCHAR limit
        data = read_file(absolute_path)

        # Handle long text chunking (aggressively reduce size)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40000,  # More aggressive splitting
            chunk_overlap=500  # Increased overlap for better context
        )
        chunks = text_splitter.split_text(data)
        
        # Validate and split oversized chunks (Hard Split)
        def validate_and_split(chunk):
            if len(chunk) > 65535:
                logger.warning(f"Chunk size {len(chunk)} exceeds VARCHAR limit. Splitting at character level.")
                return [chunk[i:i+65000] for i in range(0, len(chunk), 65000)]
            return [chunk]

        # Apply length validation to all chunks
        final_chunks = []
        for chunk in chunks:
            final_chunks.extend(validate_and_split(chunk))
        
        # Prepare documents
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": filename, "chunk_index": idx}
            )
            for idx, chunk in enumerate(final_chunks)
        ]
    
        uuids = [str(uuid4()) for _ in range(len(documents))]
        logger.info(f"Generated {len(uuids)} UUIDs for {len(documents)} documents.")

        # Batch insertion process
        batch_size = 50  # Adjust based on service limits
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = uuids[i:i+batch_size]
            try:
                connections.connect(alias="default", host="localhost", port="19530")
                collection = Collection("default_collection")
                collection.load()
                logger.info(f"Adding batch {i // batch_size + 1}...")
                
                # Prepare batch for insertion
                batch = [
                    [doc.metadata['source'] for doc in batch_docs],
                    [doc.metadata['chunk_index'] for doc in batch_docs],
                    [doc.page_content for doc in batch_docs],
                    [[0.0] * 1024 for _ in batch_docs]  # Replace with embeddings
                ]
                
                # Insert batch into Milvus
                result = collection.insert(batch)
                logger.info(f"Added batch {i // batch_size +1}.")
                
                # Flush to persist data
                collection.flush()
                entity_count = collection.num_entities
                logger.info(f"Total number of vectors in the collection 'Default Collection': {entity_count}.")
                
            except Exception as e:
                logger.error(f"Failed to add batch {i // batch_size + 1}: {e}")
                raise

        logger.info("Ingestion completed successfully.")
        return "Ingestion Successful"
    
    except Exception as e:
        vs = None
        logger.error(f"Ingestion Failed: {e}")
        raise e


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

