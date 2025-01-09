import os
import sys
import uuid
import logging
from hashlib import md5
from typing import Optional, Tuple
import boto3
import time
import tracemalloc  # Added to track memory usage

# Get the absolute path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add 'src' directory to Python path
sys.path.append(src_path)

# from src.milvus_embed import get_rag_results, process_pdf_and_store_embeddings
from src.nims_flow import ingestion, rag_results_nims
from src.openai_flow import openai_ingestion, rag_results_openai
from src.llm import chatbot
import warnings
warnings.simplefilter("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_file_hash(file) -> str:
    """
    Calculates the MD5 hash of a given file.

    Args:
        file: A file-like object to calculate the hash for.

    Returns:
        str: The hexadecimal MD5 hash of the file.
    """
    logger.debug("Calculating file hash.")
    hasher = md5()
    try:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
        file.seek(0)
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        raise
    return hasher.hexdigest()


def process_uploaded_file(uploaded_file, flag:bool) -> Optional[str]:
    """
    Processes the uploaded PDF file by saving it, generating embeddings, and storing them in Milvus.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        Optional[str]: The file path if processing is successful, otherwise None.
    """
    try:

        # Ensure the 'Data' directory exists
        data_dir = "Data"
        os.makedirs(data_dir, exist_ok=True)

        file_name = f"{uuid.uuid4().hex}.pdf"
        file_path = os.path.join(data_dir, file_name)
        logger.info(f"Saving uploaded file to {file_path}.")

        with open(file_path, "wb") as f:
            uploaded_file.seek(0)
            f.write(uploaded_file.read())

        if flag:
            ingestion(file_path)
        else:
            openai_ingestion(file_path)
        logger.info("Processing PDF and storing embeddings.")
        logger.info("File processed and embeddings stored successfully.")

        return file_path

    except Exception as e:
        logger.error(f"Error processing the file '{uploaded_file.name}': {e}")
        return None


def chatbot_response(query: str, history: list, flag:bool) -> Optional[str]:
    """
    Generates a chatbot response based on the user's query and conversation history.

    Args:
        query (str): The user's question.
        history (list): A list of tuples containing past queries and responses.

    Returns:
        Optional[str]: The chatbot's response if successful, otherwise None.
    """
    try:
        conversation_context = " ".join([f"User: {q} Bot: {a}" for q, a in history])
        logger.debug("Fetching relevant segments from Milvus.")
        final_answer = None
        if flag:
            final_answer = rag_results_nims(query)
        else: 
            relevant = rag_results_openai(query)
            logger.debug("Generating chatbot response.")
            final_answer = chatbot(query, relevant, conversation_context)
        return final_answer
    except Exception as e:
        logger.error(f"Error generating chatbot response: {e}")
        return None


def main():
    # Example user query
    user_query = input("Enter your question: ")
    
    # Benchmarking start
    start_time = time.time()
    tracemalloc.start()  # Start memory tracking

    # Example chatbot response
    chat_history = []
    response = chatbot_response(user_query, chat_history, True)  # Example: Assuming not NVIDIA model

    # Example output
    print("\nHere's my answer:")
    print(response)
    # Benchmarking end
    memory_used, _ = tracemalloc.get_traced_memory()
    elapsed_time = time.time() - start_time
    tracemalloc.stop()

    logger.info(f"Chatbot response generated in {elapsed_time:.2f} seconds.")
    logger.info(f"Memory used: {memory_used / (1024 * 1024):.2f} MB")

    # Example chat history display
    if chat_history:
        print("### Chat History")
        for i, (user_query, bot_response) in enumerate(chat_history, 1):
            print(f"Q{i}: {user_query}")
            print(f"A{i}: {bot_response}")

    print("Powered By DNN")


if __name__ == "__main__":
    main()

