import os
import uuid
import logging
from hashlib import md5
from typing import Optional, Tuple
import streamlit as st

from src.milvus_embed import get_rag_results, process_pdf_and_store_embeddings
from src.llm import chatbot
from src.nims_flow import ingestion, rag_results_nims

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


def process_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Processes the uploaded PDF file by saving it, generating embeddings, and storing them in Milvus.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        Optional[str]: The file path if processing is successful, otherwise None.
    """
    try:
        file_hash = calculate_file_hash(uploaded_file)
        if file_hash in st.session_state.uploaded_file_hashes:
            st.warning(f"The document '{uploaded_file.name}' has already been uploaded.")
            logger.info("Duplicate file upload detected.")
            return None

        # Ensure the 'Data' directory exists
        data_dir = "Data"
        os.makedirs(data_dir, exist_ok=True)

        file_name = f"{uuid.uuid4().hex}.pdf"
        file_path = os.path.join(data_dir, file_name)
        logger.info(f"Saving uploaded file to {file_path}.")

        with open(file_path, "wb") as f:
            uploaded_file.seek(0)
            f.write(uploaded_file.read())

        logger.info("Processing PDF and storing embeddings.")
        process_pdf_and_store_embeddings(file_path)
        logger.info("File processed and embeddings stored successfully.")

        st.session_state.uploaded_file_hashes[file_hash] = file_path
        return file_path

    except Exception as e:
        logger.error(f"Error processing the file '{uploaded_file.name}': {e}")
        st.error(f"Error processing the file '{uploaded_file.name}': {e}")
        return None


def chatbot_response(query: str, history: list) -> Optional[str]:
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
        relevant = get_rag_results(query)
        logger.debug("Generating chatbot response.")
        final_answer = chatbot(query, relevant, conversation_context)
        return final_answer
    except Exception as e:
        logger.error(f"Error generating chatbot response: {e}")
        st.error(f"Something went wrong while generating the response: {e}")
        return None


def initialize_session_state():
    """
    Initializes the necessary session state variables for Streamlit.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
    if "uploaded_file_hashes" not in st.session_state:
        st.session_state.uploaded_file_hashes = {}


def main():
    """
    The main function to run the Streamlit app.
    """
    # Streamlit page configuration
    st.set_page_config(
        page_title="🤖 Conversational Chatbot Q&A",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
    )

    st.title("🤖 Conversational Chatbot using Milvus")
    st.write("Upload one or more PDFs and ask me anything, and I'll try my best to provide an answer!")

    # Initialize session state variables
    initialize_session_state()

    # Dropdown for selecting LLM model
    llm_model = st.selectbox(
        "Select the LLM model:",
        options=["OpenAI GPT-4 (mini)", "NVIDIA Meta Llama3-8b"],
        index=0,
        key="llm_model"
    )

    # Dropdown for selecting Embedding model
    embedding_model = st.selectbox(
        "Select the Embedding model:",
        options=["OpenAI Embedding (text-embedding-ada-002)", "NVIDIA Embedding (nv-embedqa-e5-v5)"],
        index=0,
        key="embedding_model"
    )

    # File uploader allowing multiple files
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    processed_files = []
    if uploaded_files:
        with st.spinner("Processing the documents..."):
            for uploaded_file in uploaded_files:
                document_data_path = process_uploaded_file(uploaded_file)
                if document_data_path:
                    processed_files.append(uploaded_file.name)
                    st.session_state.uploaded_documents.append(document_data_path)
                    st.success(f"Document '{uploaded_file.name}' uploaded and processed successfully!")
                    try:
                        os.remove(document_data_path)
                        logger.info(f"Removed file {document_data_path} after processing.")
                    except Exception as e:
                        logger.warning(f"Could not remove file {document_data_path}: {e}")

    # User query input
    query = st.text_input("Enter your question here:", placeholder="Type your question...")

    if st.button("Get Answer"):
        if query.strip():
            if st.session_state.uploaded_documents:
                with st.spinner("Thinking..."):
                    # Adjust function calls based on user selections
                    if llm_model == "NVIDIA Meta Llama3-8b" and embedding_model == "NVIDIA Embedding (nv-embedqa-e5-v5)":
                        # NVIDIA model and embeddings
                        # for uploaded_file in processed_files:
                            # ingestion(uploaded_file)  # Process with NVIDIA ingestion
                        # logger.info(f"Ingestion completed in main file")
                        result = rag_results_nims(query)
                        logger.info(f"Result completed in main file")
                    else:
                        # OpenAI model and embeddings
                        for uploaded_file in processed_files:
                            process_pdf_and_store_embeddings(uploaded_file)  # Process with OpenAI embeddings
                        result = get_rag_results(query)

                    if result:
                        st.session_state.chat_history.append((query, result))
                        st.success("Here's my answer:")
                        st.write(result)
            else:
                st.warning("Please upload at least one PDF document first!")
        else:
            st.warning("Please enter a question before submitting!")

    # Display chat history
    if st.session_state.chat_history:
        st.write("### Chat History")
        for i, (user_query, bot_response) in enumerate(st.session_state.chat_history, 1):
            st.write(f"**Q{i}:** {user_query}")
            st.write(f"**A{i}:** {bot_response}")

    st.markdown("Powered By DDN")


if __name__ == "__main__":
    main()