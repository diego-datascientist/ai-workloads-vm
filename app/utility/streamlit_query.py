import streamlit as st
import time
import os
import sys
import tracemalloc
import psutil
import logging
import csv
# Update your imports as needed
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)

from src.nims_flow import rag_results_nims
# Remove if not needed anymore:
# from src.openai_flow import rag_results_openai
# from src.llm import chatbot
# from vertex_llm import gcp_vertex_llm  # Make sure gcp_vertex_llm is available here

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def performance_metrics_header(file_path):
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["Stage", "ExecutionTime(sec)"])

# Performance metrics functions
def start_performance_metrics():
    step_start_time = time.time()
    memory_before, _ = tracemalloc.get_traced_memory()
    cpu_before = psutil.cpu_percent(interval=0.1)
    return step_start_time, memory_before, cpu_before

def end_performance_metrics(step_start_time, memory_before, cpu_before, text):
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

def chatbot_response(query: str, history: list, model_choice: str) -> str:
    """
    Generates a chatbot response based on the query, chat history, and the chosen model.
    """
    step_start_time, memory_before, cpu_before = start_performance_metrics()
    try:
        # You can still build a conversation context if needed
        conversation_context = " ".join([f"User: {q} Bot: {a}" for q, a in history])
        final_answer = None

        if model_choice == "NIMS model":
            # Same logic as when flag was True
            final_answer = rag_results_nims(query)
        elif model_choice == "GCP Vertex":
            # New logic: Call the gcp_vertex_llm function
            final_answer = rag_results_nims(query, vertex=True)

        end_performance_metrics(step_start_time, memory_before, cpu_before, "Chatbot_response")
        return final_answer

    except Exception as e:
        logger.error(f"Error generating chatbot response: {e}")
        return "An error occurred while generating the response."

# Streamlit App
def main():
    st.title("Chatbot Application")
    st.write("Ask any question and the chatbot will generate an answer.")

    tracemalloc.start()
    chat_history = []

    # User input field
    user_query = st.text_input("Enter your question:")

    # Replace the checkbox with a radio button for model selection
    model_choice = st.radio(
        "Select the model to use:",
        ("NIMS model", "GCP Vertex"),
        index=0
    )

    if st.button("Submit"):
        if user_query.strip():
            st.write("Processing...")
            response = chatbot_response(user_query, chat_history, model_choice)
            st.write("### Chatbot's Answer:")
            st.write(response)

            # Append to chat history
            chat_history.append((user_query, response))

            # Display chat history
            if chat_history:
                st.write("### Chat History")
                for i, (uq, bot_response) in enumerate(chat_history, 1):
                    st.write(f"Q{i}: {uq}")
                    st.write(f"A{i}: {bot_response}")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    # file_path = "/home/daguero/idp-vm/app/utility/query_metrics.csv"
    # performance_metrics_header(file_path)
    main()