
# https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-3.2-90b-vision-instruct-maas?inv=1&invt=Abm9DQ&project=infinia-solutions-436513
# https://console.cloud.google.com/vertex-ai/colab/notebooks?inv=1&invt=Abm9DQ&project=infinia-solutions-436513&activeNb=projects%2Finfinia-solutions-436513%2Flocations%2Fus-central1%2Frepositories%2Ffc583e94-7f66-40d4-a75a-c958bece3c0b

from google.cloud import aiplatform
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import openai
from openai import OpenAI
import requests
from google.auth.transport.requests import Request
from google.oauth2 import id_token

import subprocess

# Define constants
PROJECT_ID = "infinia-solutions-436513"  # Update with your actual GCP project ID
REGION = "us-central1"

ENDPOINT = f"{REGION}-aiplatform.googleapis.com"  # Vertex AI Endpoint
MODEL = "meta/llama-3.1-405b-instruct-maas"
# IMAGE_URL = "gs://github-repo/img/gemini/intro/landmark3.jpg"
# URL = f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"


def gcp_vertex_llm(system_message, user_message):
    # Step 1: Get an access token using gcloud
    try:
        access_token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        print("Failed to get access token. Ensure you are logged into gcloud.")
        print(e)
    except Exception as e:
        print("Exception happened:", e)
        exit(1)

    # Step 2: Prepare the request headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1024,
            "temperature": 0.4,
            "top_k": 10,
            "top_p": 0.95,
        }

    # Step 4: Make the POST request
    url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"
    response = requests.post(url, headers=headers, json=payload)

    # Step 5: Handle the response
    if response.status_code == 200:
        response = response.json()['choices'][0]['message']['content']
        return response
    else:
        return response.json()

if __name__ == "__main__":
    query = "What is the difference between Machine Learning and Artificial Intelligence?"
    # Prepare the prompt
    system_message = (
        "You are an expert Conversational Chatbot. "
        "Use the context below to answer the user's question.\n\n"
        "If the context does not contain information necessary to answer the user's question, respond with: 'I don't know based on the provided context.'"
    )
    user_message = "User Question: " + query


    response = gcp_vertex_llm(system_message, user_message)
    print("ANSWER:\n",response)


# import requests
# import json
# import google.auth
# from google.auth.transport.requests import Request

# # Set your project details
# PROJECT_ID = "infinia-solutions-436513"  # Replace with your GCP project ID
# LOCATION = "us-central1"
# ENDPOINT = "https://us-central1-aiplatform.googleapis.com/v1"
# MODEL = "meta/llama-3.1-405b-instruct-maas"

# def get_access_token():
#     import subprocess
#     result = subprocess.run(
#         ["gcloud", "auth", "application-default", "print-access-token"],
#         stdout=subprocess.PIPE,
#         text=True
#     )
#     return result.stdout.strip()

# def call_vertex_ai():
#     url = f"{ENDPOINT}/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi/chat/completions"

#     headers = {
#         "Authorization": f"Bearer {get_access_token()}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": MODEL,
#         "stream": True,
#         "messages": [
#             {"role": "user", "content": "Summer travel plan to Paris"}
#         ]
#     }

#     response = requests.post(url, headers=headers, json=payload)

#     if response.status_code == 200:
#         for line in response.iter_lines():
#             if line:
#                 try:
#                     # Remove the "data: " prefix and parse the JSON
#                     line_data = line.decode("utf-8").lstrip("data: ")
#                     parsed_data = json.loads(line_data)
                    
#                     # Extract content
#                     content = parsed_data["choices"][0]["delta"].get("content")
#                     if content:
#                         print(content)
#                 except (json.JSONDecodeError, KeyError) as e:
#                     print(f"Failed to parse line: {line}. Error: {e}")
#     else:
#         print(f"Error: {response.status_code}, {response.text}")

# if __name__ == "__main__":
#     call_vertex_ai()
