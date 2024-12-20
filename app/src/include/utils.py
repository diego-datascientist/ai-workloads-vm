
import os
from typing import  Optional

from langchain_community.vectorstores import Milvus
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


from langchain.text_splitter import SentenceTransformersTokenTextSplitter

vector_store_path = "vectorstore.pkl"

TEXT_SPLITER_MODEL_HUGGINGFACE = "snowflake/arctic-embed-l"


NLIST = 128
CHUNK_SIZE = 510
METRIC_TYPE = "L2"  # Alternatives: "COSINE", "IP"
SEARCH_NPROBE = 10
CHUNK_OVERLAP = 200
INDEX_TYPE = "IVF_FLAT"

DEFAULT_MILVUS_PORT = 19530
DEFAULT_MILVUS_HOST = "milvus-standalone"
# DEFAULT_MILVUS_HOST = "localhost"


def create_vectorstore_langchain(document_embedder: "Embeddings", collection_name: str = "") -> VectorStore:
    """Create the vectorstore object for langchain based example.
    
    Args:
        document_embedder (Embeddings): Embedding model object to generate embedding of document.
        collection_name (str): The name of the collection within the vector store. Defaults to vector_db if not set.

    Returns:
        VectorStore: A VectorStore object of given vectorstore from langchain.
    """
    if not collection_name:
        collection_name = os.getenv('MILVAS_NIMS_COLLECTION_NAME', "pdf_embedding")
    
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






