import os
import logging

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter

from include.utils import create_vectorstore_langchain, get_text_splitter, get_vectorstore



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

text_splitter = None
SEARCH_LIMIT = 5
OUTPUT_FIELDS = ["text"]
EMBEDDING_MODEL_NVIDIA = "nvidia/nv-embedqa-e5-v5"

# Use RecursiveCharacterTextSplitter directly from LangChain
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_documents(filepath: str, filename: str=""):
    """Ingests documents to the VectorDB.

    Args:
        filepath (str): The path to the document file.
        filename (str): The name of the document file.

    Raises:
        ValueError: If there's an error during document ingestion or the file format is not supported.
    """
    # if not filename.endswith((".txt", ".pdf", ".md")):
    #     raise ValueError(f"{filename} is not a valid Text, PDF or Markdown file")
    try:
        # Load raw documents from the directory
        _path = filepath
        # raw_documents = UnstructuredFileLoader(_path).load()
        raw_documents = UnstructuredFileLoader("../Data/test.pdf").load()

        if raw_documents:
            global text_splitter

            # if not text_splitter:
            #     text_splitter = get_text_splitter()

            documents = text_splitter.split_documents(raw_documents)
            return documents
        else:
            logger.warning("No documents available to process!")
            return []
    except Exception as e:
        logger.error(f"Failed to Read documents due to exception {e}")
        raise ValueError("Failed to read documents. Please upload an unstructured text document.")


try:
    # initialise embedding model object
    document_embedder = NVIDIAEmbeddings(
    model=EMBEDDING_MODEL_NVIDIA,
    base_url="http://localhost:9080/v1"
)
    print("document_embedder", document_embedder)
    # create milvas connection with empty collection
    vectorstore = create_vectorstore_langchain(document_embedder=document_embedder)
    print("Vector Database", vectorstore)

    # read and split the files
    chunks = get_documents("../Data", "test.pdf")
    vs = get_vectorstore(vectorstore, document_embedder)
    
    vs.add_documents(chunks)
    logger.info(f"Ingestion completed successfully")
except Exception as e:
    vectorstore = None
    logger.info(f"Ingestion Failed: {e}")






