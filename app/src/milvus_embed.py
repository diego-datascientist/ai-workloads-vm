import os
import logging
from typing import List, Dict, Any

import numpy as np
from pymilvus import connections

# Ensure connection before using get_collection
connections.connect(alias="default", host="milvus-standalone", port="19530")

from src.milvus_db import get_collection
collection = get_collection()

from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_MILVUS_HOST = "milvus-standalone"
DEFAULT_MILVUS_PORT = 19530
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INDEX_TYPE = "IVF_FLAT"
METRIC_TYPE = "L2"  
NLIST = 128
SEARCH_NPROBE = 10
SEARCH_LIMIT = 5
OUTPUT_FIELDS = ["text"]

def split_text_recursive(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

def get_milvus_connection(host: str = DEFAULT_MILVUS_HOST, port: int = DEFAULT_MILVUS_PORT) -> None:
    try:
        connections.connect(alias="default", host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port}")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise

def process_pdf_and_store_embeddings(pdf_file_path: str) -> None:
    # Move imports inside the function to avoid circular imports
    from src.llm import get_embeddings
    from src.process_files import read_file

    logger.info(f"Processing PDF file: {pdf_file_path}")
    
    try:
        data = read_file(pdf_file_path)
        logger.debug("File read successfully.")
    except Exception as e:
        logger.error(f"Error reading file {pdf_file_path}: {e}")
        raise

    chunks = split_text_recursive(data)
    logger.info("Splitting text into chunks completed.")

    logger.info("Populating Milvus with embeddings...")
    for segment in chunks:
        try:
            embedding = get_embeddings(segment)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            collection.insert([{"text": segment, "embedding": embedding_list}])
            logger.debug(f"Inserted segment into Milvus: {segment[:30]}...")
        except Exception as e:
            logger.error(f"Error calculating embedding for segment: {segment[:30]}... Error: {e}")

    try:
        collection.flush()
        logger.info("Flushed data to Milvus.")
    except Exception as e:
        logger.error(f"Error flushing data to Milvus: {e}")
        raise

    index_params = {
        "index_type": INDEX_TYPE,
        "metric_type": METRIC_TYPE,
        "params": {"nlist": NLIST}
    }
    try:
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("Index created on 'embedding' field.")
    except Exception as e:
        logger.error(f"Error creating index on Milvus collection: {e}")
        raise

    try:
        collection.load()
        logger.info("Milvus collection loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading Milvus collection: {e}")
        raise

    logger.info("Data populated, indexed, and collection loaded in Milvus.")

def get_rag_results(query: str) -> List[Dict[str, Any]]:
    # Move imports inside the function
    from src.llm import get_embeddings

    logger.info(f"Generating embeddings for query: {query}")
    try:
        query_embedding = get_embeddings(query)
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
    except Exception as e:
        logger.error(f"Error generating embedding for query: {e}")
        raise

    search_params = {"nprobe": SEARCH_NPROBE}

    try:
        results = collection.search(
            data=[query_embedding_list],
            anns_field="embedding",
            param=search_params,
            limit=SEARCH_LIMIT,
            output_fields=OUTPUT_FIELDS
        )
        logger.debug("Search completed successfully.")
    except Exception as e:
        logger.error(f"Error during search in Milvus: {e}")
        raise

    top_matches = []
    for hit in results[0]:
        segment_text = hit.get("text")
        similarity_score = 1 - hit.distance
        top_matches.append({
            "Matching Segment": segment_text,
            "Score": similarity_score
        })
        logger.debug(f"Match found: {segment_text[:30]}... with score {similarity_score}")

    logger.info("RAG results retrieval completed.")
    return top_matches

def main():
    milvus_host = os.getenv("MILVUS_HOST", DEFAULT_MILVUS_HOST)
    milvus_port = int(os.getenv("MILVUS_PORT", DEFAULT_MILVUS_PORT))
    get_milvus_connection(host=milvus_host, port=milvus_port)

if __name__ == "__main__":
    main()