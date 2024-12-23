import os
import logging
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from langchain_nvidia_ai_endpoints import  NVIDIAEmbeddings, ChatNVIDIA

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Retrieve OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in environment variables.")
    raise EnvironmentError("OPENAI_API_KEY is not set.")

# Set OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

client_nims = ChatNVIDIA(
                base_url=f"http://nemollm-inference:8000/v1",
                temperature=0,
                top_p=1,
                max_tokens=1024,
            )

# Constants for models
CHAT_MODEL = "gpt-4o-mini"
CHAT_MODEL_NIMS = "meta/llama3-70b-instruct"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_MODEL_NVIDIA = "nvidia/nv-embedqa-e5-v5"


def chatbot(question: str, context: str, history: str) -> str:
    """
    Generates a chatbot response based on the user's question, provided context, and conversation history.

    Args:
        question (str): The user's question.
        context (str): Relevant context to assist in answering the question.
        history (str): The conversation history.

    Returns:
        str: The chatbot's response.
    """
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

def get_embeddings(text: str) -> Any:
    """
    Generates embeddings for the given text using OpenAI's embedding model.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        Any: The embedding vector.
    """
    try:
        response = client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        embedding = response.data[0].embedding
        logger.info("Embeddings generated successfully.")
        return embedding
    except Exception as e:
        logger.error(f"Unexpected error in get_embeddings function: {e}")
        raise
