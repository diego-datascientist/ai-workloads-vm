
import os
from openai import OpenAI
from typing import  Optional
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Milvus
from langchain_core.vectorstores import VectorStore

from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_openai import OpenAIEmbeddings

NLIST = 128
CHUNK_SIZE = 510
METRIC_TYPE = "L2"  # Alternatives: "COSINE", "IP"
SEARCH_NPROBE = 10
CHUNK_OVERLAP = 200
INDEX_TYPE = "IVF_FLAT"

TEXT_SPLITER_MODEL_HUGGINGFACE = "snowflake/arctic-embed-l"

DEFAULT_MILVUS_PORT = 19530
# DEFAULT_MILVUS_HOST = "milvus-standalone"
DEFAULT_MILVUS_HOST = "localhost"



def get_embedder(model_name:str):
    document_embedder = NVIDIAEmbeddings(
            model=model_name,
            base_url="http://localhost:9080/v1"  
            # base_url="http://nemollm-embedding:8000/v1"  
        )
    print("document_embedder initialized successfully.")
    return document_embedder

def get_embedder_openai():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set in environment variables.")
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings



def get_chat_model():
    return ChatNVIDIA(
        base_url="http://localhost:8000/v1", 
        # base_url="http://nemollm-inference:8000/v1", 
        temperature=0,
        top_p=1,
        max_tokens=1024,
    )

def get_vertex_model():
    ## directly being used
    pass



def get_openai_chat_model():
    # Retrieve OpenAI API key from environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set in environment variables.")
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    # Set OpenAI API key
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client

def get_ranking_model():
    return NVIDIARerank( #base_url=f"http://ranking-ms:8000/v1",
                        base_url=f"http://localhost:1976/v1",
                        top_n=10,
                        truncate="END")


def create_vectorstore_langchain(document_embedder: "Embeddings", collection_name: str = "") -> VectorStore:
    """Create the vectorstore object for langchain based example.
    
    Args:
        document_embedder (Embeddings): Embedding model object to generate embedding of document.
        collection_name (str): The name of the collection within the vector store. Defaults to vector_db if not set.

    Returns:
        VectorStore: A VectorStore object of given vectorstore from langchain.
    """
    if not collection_name:
        collection_name = os.getenv('MILVAS_NIMS_COLLECTION_NAME', "default_collection")
    
    vectorstore = Milvus(
        document_embedder,
        connection_args={"host": DEFAULT_MILVUS_HOST, "port": DEFAULT_MILVUS_PORT},
        collection_name=collection_name,
        index_params={"index_type":INDEX_TYPE , "metric_type": METRIC_TYPE, "nlist": NLIST},
        search_params={"nprobe": SEARCH_NPROBE},
        auto_id = True
    )
    return vectorstore


def get_text_splitter() -> SentenceTransformersTokenTextSplitter:
    """Return the token text splitter instance from langchain.
    
    Returns:
        SentenceTransformersTokenTextSplitter: Splitting text to tokens using sentence model tokenizer
    """

    return SentenceTransformersTokenTextSplitter(
        model_name=TEXT_SPLITER_MODEL_HUGGINGFACE,
        tokens_per_chunk=CHUNK_SIZE - 2,
        chunk_overlap=CHUNK_OVERLAP,
    )


def get_vectorstore(vectorstore: Optional["VectorStore"], document_embedder: "Embeddings") -> VectorStore:
    """Retrieves or creates a VectorStore object from langchain.

    Args:
        vectorstore (Optional[VectorStore]): VectorStore object from langchain.
        document_embedder (Embeddings): Embedding model object to generate embedding of document.

    Returns:
        VectorStore: A VectorStore object of given vectorstore from langchain.
    """
    if vectorstore is None:
        return create_vectorstore_langchain(document_embedder)
    return vectorstore






