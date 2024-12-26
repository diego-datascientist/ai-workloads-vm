import os
import logging

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


from .process_files import read_file
from .include.utils import  get_vectorstore , get_embedder_openai, get_openai_chat_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Use RecursiveCharacterTextSplitter directly from LangChain
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

SEARCH_LIMIT = 5
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
vs = None
k = 10

# Embedding Object
document_embedder = get_embedder_openai()
logger.info("OpenAI Embedding Object initialized successfully.")

# Milvus Database Object
vs = get_vectorstore(vs, document_embedder)

logger.info("Vector Database initialized.")


def openai_ingestion(filename:str, filepath:str="../Data"):
    try:
        absolute_path = os.path.abspath(filename)
        print("Absolute path in get_documents:", absolute_path)
        print("Does file exist:", os.path.exists(absolute_path))
        data = read_file(absolute_path)

        chunks = text_splitter.split_text(data)

        global vs
        vs = get_vectorstore(vs, document_embedder)
        
        documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
        vs.add_documents(documents)
        logger.info("Ingestion completed successfully.")
        return "Ingestion Successful"
    except Exception as e:
        vs = None
        logger.info(f"Ingestion Failed: {e}")


def openai_query(question: str, context:str, history:str=[]):
    global vs
    if vs is None:
        vs = get_vectorstore(vs, document_embedder)

    similar_docs = vs.similarity_search(question, k)

    context = "\n\n".join([doc.page_content for doc in similar_docs])

    client = get_openai_chat_model()
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Conversational Chatbot assistant for DDN company USA. "
                    "Your objective is to answer the user's questions using the provided context inside this delimiter ####. "
                    "Use the conversation history if the user asks anything from its previous chat; this history is mentioned inside this delimiter @@@@. "
                    "In case of any confusion, counter question the user for clarity regarding the subject matter."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User Question: {question}\n"
                    f"Relevant Context to find the answer from: #### {context} ####\n"
                    f"Chat History: @@@@ {history} @@@@."
                )
            }
        ]

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        logger.info("Chatbot response generated successfully.")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Unexpected error in chatbot function: {e}")
        raise


def rag_results_openai(query:str):
    try:
        answer = openai_query(query)
        return answer
    except Exception as e:
        logger.info(f"Failed to Perform NIM RAG: {e}")


