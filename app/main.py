import os
import uuid
import logging
from hashlib import md5
from typing import Optional, Tuple
import streamlit as st
import boto3

# from src.milvus_embed import get_rag_results, process_pdf_and_store_embeddings
from src.nims_flow import ingestion, rag_results_nims
from src.openai_flow import openai_ingestion, rag_results_openai
from src.llm import chatbot


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

        if flag:
            ingestion(file_path)
        else:
            openai_ingestion(file_path)
        logger.info("Processing PDF and storing embeddings.")
        logger.info("File processed and embeddings stored successfully.")

        st.session_state.uploaded_file_hashes[file_hash] = file_path
        return file_path

    except Exception as e:
        logger.error(f"Error processing the file '{uploaded_file.name}': {e}")
        st.error(f"Error processing the file '{uploaded_file.name}': {e}")
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
        st.error(f"Something went wrong while generating the response: {e}")
        return None


# def process_s3_file(cred:str, s3_path: str, flag: bool) -> Optional[str]:
#     try:
#         bucket, key = s3_path.replace("s3://", "").split('/', 1)
#         local_path = f"Data/{uuid.uuid4().hex}.pdf"
#         os.makedirs("Data", exist_ok=True)
#         download_file(cred, bucket, key, local_path)
#         logger.info(f"Downloaded file from S3 to {local_path}")

#         if flag:
#             ingestion(local_path)
#         else:
#             openai_ingestion(local_path)

#         return local_path

#     except Exception as e:
#         st.error(f"Error downloading file from S3: {e}")
#         return None


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

    upload_type = st.selectbox(
        "Select Upload Type:",
        options=["Direct_Upload", "S3_Upload"]
    )
    if upload_type == "Direct_Upload":
        # File uploader allowing multiple files
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Processing the documents..."):
                for uploaded_file in uploaded_files:
                    document_data_path = None
                    if embedding_model == "NVIDIA Embedding (nv-embedqa-e5-v5)":
                        document_data_path = process_uploaded_file(uploaded_file, True)
                    else:
                        document_data_path = process_uploaded_file(uploaded_file, False)

                    if document_data_path:
                        st.session_state.uploaded_documents.append(document_data_path)
                        st.success(f"Document '{uploaded_file.name}' uploaded and processed successfully!")
                        try:
                            os.remove(document_data_path)
                            logger.info(f"Removed file {document_data_path} after processing.")
                        except Exception as e:
                            logger.warning(f"Could not remove file {document_data_path}: {e}")
    # else:
    #     accessKey = st.text_input("Please Enter AWS Access Key!)")
    #     secretAccessKey = st.text_input("Please Enter AWS Secret Access Key!)")
    #     s3_path = st.text_input("Enter S3 Path (e.g., s3://bucket-name/file.pdf)")
    #     if st.button("Fetch from S3"):
    #         with st.spinner("Fetching and processing..."):
    #             if s3_path:
    #                 if embedding_model == "NVIDIA Embedding":
    #                     process_s3_file({accessKey, secretAccessKey}, s3_path, True)
    #                 else:
    #                     process_s3_file({accessKey, secretAccessKey}, s3_path, False)
                    st.success("File fetched and processed!")
    # User query input
    query = st.text_input("Enter your question here:", placeholder="Type your question...")

    if st.button("Get Answer"):
        if query.strip():
            # if st.session_state.uploaded_documents:
            with st.spinner("Thinking..."):
                response= None
                if llm_model == "NVIDIA Meta Llama3-8b":
                    response = chatbot_response(query, st.session_state.chat_history, True)
                else: 
                    response = chatbot_response(query, st.session_state.chat_history, False)
                if response:
                    st.session_state.chat_history.append((query, response))
                    st.success("Here's my answer:")
                    st.write(response)
            # else:
                # st.warning("Please upload at least one PDF document first!")
        else:
            st.warning("Please enter a question before submitting!")

    # Display chat history
    if st.session_state.chat_history:
        st.write("### Chat History")
        for i, (user_query, bot_response) in enumerate(st.session_state.chat_history, 1):
            st.write(f"**Q{i}:** {user_query}")
            st.write(f"**A{i}:** {bot_response}")

    st.markdown("Powered By DNN")


if __name__ == "__main__":
    main()