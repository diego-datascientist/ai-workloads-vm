# 🤖 Conversational Chatbot with Milvus and OpenAI

Welcome to the **Conversational Chatbot with Milvus and OpenAI** project! This application leverages the power of OpenAI's language models and Milvus, a high-performance vector database, to provide intelligent, context-aware responses based on uploaded PDF documents. The user-friendly interface is built with Streamlit, allowing seamless interaction and document processing.

## Table of Contents

- [🔍 Project Overview](#-project-overview)
- [🚀 Features](#-features)
- [🛠️ Technologies Used](#%EF%B8%8F-technologies-used)
- [📦 Architecture](#-architecture)
- [⚙️ Prerequisites](#%EF%B8%8F-prerequisites)
- [📥 Installation](#%EF%B8%8F-installation)
- [🖥️ Running the Application](#%EF%B8%8F-running-the-application)
- [🔧 Configuration](#-configuration)
- [📝 Usage](#-usage)
- [🐞 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📫 Contact](#-contact)

## 🔍 Project Overview

The **Conversational Chatbot with Milvus and OpenAI** enables users to:

1. **Upload PDF Documents:** Users can upload one or multiple PDF files containing relevant information.
2. **Process and Embed Content:** The application processes the PDFs, extracts text, splits it into manageable chunks, and generates embeddings using OpenAI's embedding models.
3. **Store Embeddings in Milvus:** These embeddings are stored in Milvus, allowing efficient similarity search and retrieval.
4. **Interactive Chat Interface:** Users can ask questions related to the uploaded documents. The chatbot leverages the stored embeddings to provide accurate and context-aware responses.

This setup ensures that the chatbot not only understands the queries but also references the specific content from the uploaded documents, offering precise and relevant answers.

## 🚀 Features

- **PDF Upload and Processing:** Seamlessly upload PDFs and extract meaningful content.
- **Embeddings Generation:** Utilize OpenAI's advanced models to generate high-quality text embeddings.
- **Efficient Storage with Milvus:** Store and manage embeddings in Milvus for rapid retrieval and similarity searches.
- **Interactive Streamlit Interface:** User-friendly interface for uploading documents and interacting with the chatbot.
- **Context-Aware Responses:** The chatbot references uploaded documents to provide accurate answers.

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive web application.
- **[OpenAI API](https://openai.com/api/):** To generate text embeddings and facilitate the chatbot's language capabilities.
- **[Milvus](https://milvus.io/):** A vector database optimized for storing and retrieving high-dimensional embeddings.
- **[Python](https://www.python.org/):** The primary programming language used.
- **[Docker & Docker Compose](https://www.docker.com/):** For containerizing and orchestrating the application and its dependencies.
- **[PyMilvus](https://pymilvus.readthedocs.io/):** Python SDK for interacting with Milvus.
- **[Python-dotenv](https://pypi.org/project/python-dotenv/):** For managing environment variables.

## 📦 Architecture

1. **Streamlit Frontend:**
   - Handles user interactions, file uploads, and displays chatbot responses.
2. **Backend Services:**
   - **Milvus:** Stores embeddings for efficient similarity searches.
   - **OpenAI API:** Generates embeddings and processes natural language queries.
3. **Docker Compose:**
   - Orchestrates the deployment of Milvus and the Streamlit application, ensuring all services run smoothly together.

## ⚙️ Prerequisites

Before setting up the project, ensure you have the following installed on your system:

- **[Docker](https://www.docker.com/get-started):** Version 20.10 or higher.
- **[Docker Compose](https://docs.docker.com/compose/install/):** Version 1.29 or higher.
- **[Git](https://git-scm.com/downloads):** For cloning the repository.

Additionally, you'll need:

- **OpenAI API Key:** Sign up at [OpenAI](https://openai.com/) to obtain your API key.
- **Environment Variables:** Ensure you can set environment variables on your system or use a `.env` file.

## 📥 Installation

### 1. Clone the Repository

```bash
git clone https://github.red.datadirectnet.com/red/ai-workload.git
cd conversational-chatbot
```

### 2. Set Up Environment Variables
Create a .env file in the root directory and add the following variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
MILVUS_HOST=milvus-standalone
MILVUS_PORT=19530
```

### 3. Build and Run with Docker Compose
Ensure Docker and Docker Compose are installed and running on your system.
```bash
docker-compose up --build
```

This command will:
	•	Build the Streamlit Application Image.
	•	Pull and Run the Milvus Container.
	•	Set Up Networking Between Containers.

## 🖥️ Running the Application

Once Docker Compose has successfully built and started the containers, access the Streamlit application by navigating to:
```bash
http://localhost:8501
```

## 🐞 Troubleshooting

### Common Issues and Solutions

1. **`'function' object has no attribute 'flush'` Error:**
   - **Cause:** Assigning a function to the `collection` variable instead of the `Collection` object.
   - **Solution:** Ensure you call the `get_collection()` function when assigning to `collection`.
     ```python
     from milvus_setup import get_collection
     collection = get_collection()  # Correct: Calls the function to get the Collection object
     ```

2. **Milvus Connection Errors:**
   - **Cause:** Milvus server is not running or incorrect host/port configurations.
   - **Solution:** Verify that Milvus is running using Docker Compose and that the `MILVUS_HOST` and `MILVUS_PORT` are correctly set in the `.env` file.

3. **OpenAI API Errors:**
   - **Cause:** Invalid or missing OpenAI API key.
   - **Solution:** Ensure the `OPENAI_API_KEY` is correctly set in the `.env` file.

4. **Docker Compose Issues:**
   - **Cause:** Conflicting ports or insufficient permissions.
   - **Solution:** Check that the required ports (`19530`, `19121`, `8501`) are free and that Docker has the necessary permissions to bind to these ports.

### Additional Debugging Steps

- **Check Logs:**
  - Use Docker logs to inspect the output of both the Milvus and application containers.
    ```bash
    docker logs milvus
    ```

- **Verify Environment Variables:**
  - Ensure that all required environment variables are correctly set and accessible within the Docker containers.

- **Test Individual Components:**
  - Before running the full application, test Milvus connectivity and OpenAI API interactions separately to isolate issues.