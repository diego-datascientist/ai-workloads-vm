import os
import logging
from dotenv import load_dotenv
from operator import itemgetter

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.runnables.passthrough import RunnableAssign  # Ensure correct import path

from .process_files import read_file
from .include.utils import (
    create_vectorstore_langchain,
    get_vectorstore,
    get_embedder,
    get_ranking_model,
    get_chat_model
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

SEARCH_LIMIT = 50  # Adjusted based on reference code
TOP_K = 5  # Number of top documents to use
EMBEDDING_MODEL_NVIDIA = "nvidia/nv-embedqa-e5-v5"
vs = None

# Initialize Embedding Object
document_embedder = get_embedder(EMBEDDING_MODEL_NVIDIA)
logger.info("Document embedder initialized successfully.")

# Initialize Milvus Database Object
vs = get_vectorstore(vs, document_embedder=document_embedder)
logger.info("Vector Database initialized.")


# Initialize the Reranker
ranker = get_ranking_model()

logger.info("Reranker initialized successfully.")


def ingestion(filename: str, filepath: str = "../Data"):
    try:
        absolute_path = os.path.abspath(os.path.join(filepath, filename))
        logger.info(f"Absolute path in get_documents: {absolute_path}")
        logger.info(f"Does file exist: {os.path.exists(absolute_path)}")

        data = read_file(absolute_path)
        chunks = text_splitter.split_text(data)

        global vs
        vs = get_vectorstore(vs, document_embedder=document_embedder)

        # Assigning metadata (e.g., source and chunk index)
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": filename, "chunk_index": idx}
            )
            for idx, chunk in enumerate(chunks)
        ]
        vs.add_documents(documents)
        logger.info("Ingestion completed successfully.")
        return "Ingestion Successful"
    except Exception as e:
        vs = None
        logger.error(f"Ingestion Failed: {e}")
        raise e  # Re-raise the exception for upstream handling



def re_rank(q):
    context_chain = RunnableAssign(
        {"context": itemgetter("input") | vs.as_retriever(search_kwargs={"k": 40})}
    )
    if ranker:
        logger.info(
            f"Narrowing the collection from 40 results and further narrowing it to 3 with the reranker."
        )
        context_reranker = RunnableAssign(
            {
                "context": lambda input: ranker.compress_documents(
                    query=input['input'], documents=input['context']
                )
            }
        )

        retrieval_chain = context_chain | context_reranker
    else:
        retrieval_chain = context_chain 

    # Handling Retrieval failure
    docs = retrieval_chain.invoke({"input": q})
    print("DOCS", docs)
    if not docs:
        logger.warning("Retrieval failed to get any relevant context")
        return iter(
            [
                "No response generated from LLM, make sure your query is relavent to the ingested document."
            ]
        )

    logger.debug(f"Retrieved documents are: {docs}")



def answer_query(query: str):
    global vs
    if vs is None:
        vs = create_vectorstore_langchain(document_embedder=document_embedder)
        logger.info("Vector store recreated as it was None.")




    # Initial similarity search
    initial_docs = vs.similarity_search(query, SEARCH_LIMIT)
    logger.info(f"Initial retrieved {len(initial_docs)} documents for query: '{query}'")

    # if ranker:
    #     re_rank(query)

    # Rerank the initial documents using the correct method
    try:
        reranked_docs = ranker.compress_documents(query=query, documents=initial_docs)
        logger.info(f"Reranked {len(reranked_docs)} documents.")
    except AttributeError as ae:
        logger.error(f"Reranking failed: {ae}")
        reranked_docs = initial_docs  # Fallback to initial docs if reranking fails
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        reranked_docs = initial_docs  # Fallback to initial docs if reranking fails

    # Select top-k documents after reranking
    print(f"there were {len(reranked_docs)} documents now selecting {TOP_K} from it")
    top_k_docs = reranked_docs[:TOP_K]
    context = "\n\n".join([doc.page_content for doc in top_k_docs])
    print("CONEXT", context)
    logger.debug(f"Context for LLM: {context}")

    # Prepare the prompt
    system_message = (
        "You are an expert Conversational Chatbot. "
        "Use the context below to answer the user's question. If there's any confusion, ask clarifying questions.\n\n"
        "If you don't know something, just say 'I don't know'.\n\n"
        "Context:\n" + context
    )
    user_message = "User Question: " + query

    inference_client = get_chat_model()

    # Generate response using the local inference model
    try:
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
        logger.info("Response generated successfully.")
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        full_response = "I'm sorry, I couldn't process your request at the moment."

    return full_response.strip()

def rag_results_nims(query: str):
    logger.info("Received query for RAG processing.")
    try:
        answer = answer_query(query)
        logger.info("RAG processing completed successfully.")
        return answer
    except Exception as e:
        logger.error(f"Failed to Perform NIM RAG: {e}")
        return "I'm sorry, something went wrong while processing your request."

