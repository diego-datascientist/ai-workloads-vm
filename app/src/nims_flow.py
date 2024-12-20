import os
import logging

# from langchain_community.document_loaders import UnstructuredFileLoader
# from langchain_unstructured import UnstructuredLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
## ranker 
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnableAssign

from .process_files import read_file

# from include.utils import create_vectorstore_langchain, get_vectorstore

from .include.utils import create_vectorstore_langchain, get_vectorstore
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
OUTPUT_FIELDS = ["text"]
EMBEDDING_MODEL_NVIDIA = "nvidia/nv-embedqa-e5-v5"
vs = None
k = 3

document_embedder = NVIDIAEmbeddings(
            model=EMBEDDING_MODEL_NVIDIA,
            # base_url="http://localhost:9080/v1"  
            base_url="http://nemollm-embedding:8000/v1"  
        )
logger.info("document_embedder initialized successfully.")



vs = create_vectorstore_langchain(document_embedder=document_embedder)

logger.info("Vector Database initialized.")

ranker = NVIDIARerank( base_url=f"http://ranking-ms:8000/v1", top_n=4, truncate="END")
logger.info("Ranker initialized successfully.")

def ingestion(filename:str, filepath:str="../Data"):

    try:
        absolute_path = os.path.abspath(filename)
        print("Absolute path in get_documents:", absolute_path)
        print("Does file exist:", os.path.exists(absolute_path))
        data = read_file(absolute_path)

        chunks = text_splitter.split_text(data)

        print("Chunks", chunks)
        global vs
        vs = get_vectorstore(vs, document_embedder)
        
        documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
        vs.add_documents(documents)
        logger.info("Ingestion completed successfully.")
        return "Ingestion Successful"
    except Exception as e:
        vectorstore = None
        logger.info(f"Ingestion Failed: {e}")


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

    # ranker check 
    k = 40 if ranker else 3
    print("K", k)
    if ranker:
        re_rank(query)

    
    similar_docs = vs.similarity_search(query, k)

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
        # base_url="http://localhost:8000/v1", 
        base_url="http://nemollm-inference:8000/v1", 
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




# if __name__ == "__main__":
#     ingestion("test.pdf")
#     # query 
#     result = rag_results_nims("tell me about DDN.")
#     print("result", result)
