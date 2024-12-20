import os
import logging

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter

from include.utils import create_vectorstore_langchain, get_vectorstore



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
vs = None

document_embedder = NVIDIAEmbeddings(
        model=EMBEDDING_MODEL_NVIDIA,
        base_url="http://localhost:9080/v1"
    )

vs = create_vectorstore_langchain(document_embedder=document_embedder)

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


def ingestion():
    try:                
        chunks = get_documents("../Data", "test.pdf")
        vs = get_vectorstore(vs, document_embedder)
        
        vs.add_documents(chunks)
        logger.info(f"Ingestion completed successfully")
    except Exception as e:
        vs = None
        logger.info(f"Ingestion Failed: {e}")




def answer_query(query: str):


    global vs
    if vs is None:
        vs = create_vectorstore_langchain(document_embedder=document_embedder)

    similar_docs = vs.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    print("Context", context)
    # Prepare the prompt
    system_message = (
        "You are an expert Conversational Chatbot"
        "Use the context below to answer the user's question. If there's any confusion, ask clarifying questions.\n\n"
        "If you don't know something just say i don't know"
        "Context:\n" + context
    )
    user_message = "User Question: " + query
    

    inference_client = ChatNVIDIA(
        base_url="http://localhost:8000/v1", 
        temperature=0,
        top_p=1,
        max_tokens=1024,
    )

# Generate response using the local inference model
    response = inference_client.invoke(
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    full_response = response.content

    return full_response.strip()


def rag_results_nims(query:str):
    print(" i am in query nims funciton")
    try:
        answer = answer_query(query)
        return answer
    except Exception as e:
        logger.info(f"Failed to Perform NIM RAG: {e}")




if __name__ == "__main__":
    ingestion()
    # query 
    result = rag_results_nims("tell me about DDN.")
    print("result", result)


