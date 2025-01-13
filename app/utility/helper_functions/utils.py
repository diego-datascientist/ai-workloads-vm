
import os
import logging
from typing import  Optional
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Milvus
from langchain_core.vectorstores import VectorStore

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_nvidia_ai_endpoints import  NVIDIAEmbeddings
from langchain_openai import OpenAIEmbeddings
import warnings
warnings.simplefilter("ignore", category=UserWarning)


NLIST = 128
CHUNK_SIZE = 510
METRIC_TYPE = "L2"
SEARCH_NPROBE = 10
CHUNK_OVERLAP = 200
INDEX_TYPE = "IVF_FLAT"

TEXT_SPLITER_MODEL_HUGGINGFACE = "snowflake/arctic-embed-l"

DEFAULT_MILVUS_PORT = 19530
# DEFAULT_MILVUS_HOST = "milvus-standalone-01"
DEFAULT_MILVUS_HOST = "localhost"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



def get_embedder(model_name:str):
    document_embedder = NVIDIAEmbeddings(
            model=model_name,
            base_url="http://localhost:9080/v1"  
            # base_url="http://nemollm-embedding:8000/v1"  
        )
    return document_embedder

def get_embedder_openai():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set in environment variables.")
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings

def create_vectorstore_langchain(document_embedder: "Embeddings", collection_name: str = "") -> VectorStore:
    if not collection_name:
        collection_name = os.getenv('MILVAS_NIMS_COLLECTION_NAME', "default_collection")
    try:
        vectorstore = Milvus(
            document_embedder,
            connection_args={"host": DEFAULT_MILVUS_HOST, "port": DEFAULT_MILVUS_PORT},
            collection_name=collection_name,
            index_params={"index_type":INDEX_TYPE , "metric_type": METRIC_TYPE, "nlist": NLIST},
            search_params={"nprobe": SEARCH_NPROBE},
            auto_id = True
        )
        logger.info("VectorStore iniliazed successfully!")
    except Exception as e:
        logger.error(f"VectorStore Error: {e}.")
    return vectorstore


def get_text_splitter() -> SentenceTransformersTokenTextSplitter:
    return SentenceTransformersTokenTextSplitter(
        model_name=TEXT_SPLITER_MODEL_HUGGINGFACE,
        tokens_per_chunk=CHUNK_SIZE - 2,
        chunk_overlap=CHUNK_OVERLAP,
    )


def get_vectorstore(vectorstore: Optional["VectorStore"], document_embedder: "Embeddings") -> VectorStore:
    if vectorstore is None:
        return create_vectorstore_langchain(document_embedder)
    return vectorstore
