import os
import sys
import uuid
import logging
from hashlib import md5
from typing import Optional, Tuple
import boto3
import time, csv
import tracemalloc
import psutil  # Added for CPU usage

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

from src.nims_flow import ingestion, rag_results_nims
from src.openai_flow import openai_ingestion, rag_results_openai
from src.llm import chatbot
import warnings
warnings.simplefilter("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def performance_metrics_header(file_path):
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Stage", "ExecutionTime(sec)", "MemoryUsage(MB)", "CPU-Usage(%)"])
def start_performance_metrics():
    # ------------ Performance metrics start ------------------
    step_start_time = time.time()
    memory_before, _ = tracemalloc.get_traced_memory()
    cpu_before = psutil.cpu_percent(interval=15.0)
    return step_start_time, memory_before, cpu_before

def end_performance_metrics(step_start_time, memory_before, cpu_before, text, file_path):
    # ------------ Performance metrics end ------------------
    cpu_after = psutil.cpu_percent(interval=None)
    memory_after, _ = tracemalloc.get_traced_memory()
    step_end_time = time.time()

    elapsed_time = step_end_time - step_start_time
    memory_diff = (memory_after - memory_before) / (1024 * 1024)
    cpu_usage = cpu_after

    # Log to console/file using logger
    logger.info(
        f"[{text}] Time: {elapsed_time:.4f}s, "
        f"Memory Diff: {memory_diff:.4f} MB, "
        f"CPU usage diff: {cpu_usage:.2f}%"
    )

    # Append the data to the CSV file
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            text,
            f"{elapsed_time:.4f}",
            # f"{memory_diff:.4f}",
            # f"{cpu_usage:.2f}"
        ])



def chatbot_response(query: str, history: list, flag: bool) -> Optional[str]:
    """Generates a chatbot response based on the query and chat history."""
    file_path = "/home/daguero/idp-vm/app/utility/query_metrics.csv"
    # ------------ Performance metrics start ------------------
    step_start_time, memory_before, cpu_before = start_performance_metrics()

    try:
        conversation_context = " ".join([f"User: {q} Bot: {a}" for q, a in history])
        logger.debug("Fetching relevant segments from Milvus.")
        final_answer = None

        if flag:
            final_answer = rag_results_nims(query, True)
        else:
            relevant = rag_results_openai(query)
            logger.debug("Generating chatbot response.")
            final_answer = chatbot(query, relevant, conversation_context)

        # ------------ Performance metrics end ------------------
        end_performance_metrics(step_start_time, memory_before, cpu_before, "Chatbot_response", file_path)
        # -------------------------------------------------------

        return final_answer

    except Exception as e:
        logger.error(f"Error generating chatbot response: {e}")
        return None


def main():
    """Main entry point to test chatbot response via a console input."""
    file_path = "/home/daguero/idp-vm/app/utility/query_metrics.csv"
    user_query = input("Enter your question: ")
    performance_metrics_header(file_path)
    tracemalloc.start()

    chat_history = []
    response = chatbot_response(user_query, chat_history, True)

    print("\nHere's my answer:")
    print(response)
    
    # Stop memory tracking
    tracemalloc.stop()

    if chat_history:
        print("### Chat History")
        for i, (uq, bot_response) in enumerate(chat_history, 1):
            print(f"Q{i}: {uq}")
            print(f"A{i}: {bot_response}")

    print("Powered By DNN")


if __name__ == "__main__":
    main()