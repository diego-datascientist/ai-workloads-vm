import os
import logging
from typing import Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    exceptions as milvus_exceptions,
    utility,
    connections
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEFAULT_MILVUS_HOST = "milvus-standalone"
DEFAULT_MILVUS_PORT = 19530
COLLECTION_NAME = "pdf_embeddings"
VECTOR_DIMENSION = 1536
VARCHAR_MAX_LENGTH = 65535
PRIMARY_KEY_FIELD = "id"
TEXT_FIELD = "text"
EMBEDDING_FIELD = "embedding"

_collection: Optional[Collection] = None

def get_env_variable(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        logger.error(f"Environment variable '{key}' is not set and no default value provided.")
        raise EnvironmentError(f"Environment variable '{key}' is required but not set.")
    return value

def ensure_connection():
    """
    Ensure that the default connection is established. 
    If not connected, raise an error indicating that the calling code must connect first.
    """
    if not connections.has_connection(alias="default"):
        logger.error("No active connection to Milvus found. Please connect before calling get_collection or setup_collection.")
        raise milvus_exceptions.ConnectionException("Not connected to Milvus. Call connections.connect() before setup_collection or get_collection.")

def drop_existing_collection(collection_name: str) -> None:
    """
    Drops the existing Milvus collection if it exists.
    """
    ensure_connection()
    try:
        if utility.has_collection(collection_name):
            existing_collection = Collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' exists. Dropping it for a fresh start...")
            existing_collection.drop()
            logger.info(f"Collection '{collection_name}' dropped successfully.")
        else:
            logger.info(f"Collection '{collection_name}' does not exist. No need to drop.")
    except milvus_exceptions.MilvusException as e:
        logger.error(f"Error while dropping collection '{collection_name}': {e}")
        raise

def create_collection(collection_name: str) -> Collection:
    """
    Creates a new Milvus collection with the specified schema.
    """
    ensure_connection()
    fields = [
        FieldSchema(name=PRIMARY_KEY_FIELD, dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name=TEXT_FIELD, dtype=DataType.VARCHAR, max_length=VARCHAR_MAX_LENGTH),
        FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION),
    ]
    schema = CollectionSchema(fields=fields, description="A collection of text and embeddings")

    try:
        new_collection = Collection(name=collection_name, schema=schema)
        logger.info(f"New collection '{collection_name}' created successfully.")
        return new_collection
    except milvus_exceptions.MilvusException as e:
        logger.error(f"Error while creating collection '{collection_name}': {e}")
        raise

def setup_collection() -> None:
    """
    Sets up the Milvus collection. Assumes connection is already established externally.
    """
    global _collection
    ensure_connection()
    drop_existing_collection(COLLECTION_NAME)
    _collection = create_collection(COLLECTION_NAME)

def get_collection() -> Collection:
    global _collection
    if _collection is None:
        logger.info("Collection not initialized. Setting up the collection.")
        setup_collection()
    return _collection

def main():
    try:
        setup_collection()
    except Exception as e:
        logger.error(f"Failed to set up the Milvus collection: {e}")
        exit(1)

# if __name__ == "__main__":
#     main()