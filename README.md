# Azure-Data-Ingestion Tool

# Bakasura Knowledge Devourer

**Bakasura Knowledge Devourer** is a powerful document ingestion system designed to extract, process, and store knowledge from PDF documents. Inspired by the insatiable mythological figure, this application "devours" documents, making their content ready for advanced search and retrieval through semantic understanding.

## 🚀 Features

* **Multi-File PDF Upload:** Ingest multiple PDF documents simultaneously through a user-friendly drag-and-drop interface.
* **Advanced Content Extraction:** Goes beyond simple text extraction to process complex elements within documents.
    * **Standard Text:** Pulls all textual content from PDFs.
    * **Table Recognition:** Identifies and extracts tabular data.
    * **OCR for Images:** Uses Azure AI Vision to perform Optical Character Recognition (OCR) on images within the PDFs, ensuring no information is lost.
* **Semantic Embedding Generation:** Leverages **Azure OpenAI** to convert text chunks into meaningful vector embeddings, capturing the semantic context of the information.
* **Vector Storage & Indexing:** Stores the generated embeddings in **Azure Cosmos DB for MongoDB (vCore)**, creating a searchable and scalable knowledge base.
* **System Diagnostics:** Includes built-in tools to test the connection to Azure OpenAI, ensuring the core services are operational.

## 🛠️ Technology Stack

* **Backend:** Python
* **Frontend:** Streamlit
* **Containerization:** Docker
* **AI & Embeddings:** Azure OpenAI
* **Image & Text Recognition:** Azure AI Vision
* **Database:** Azure Cosmos DB for MongoDB (vCore)

## ⚙️ Setup and Installation

Follow these instructions to get the application running on your local machine or as a Docker container.

### Recommended Approach: Docker

For the most reliable and consistent experience, **it is highly recommended to run this application using Docker**. This approach encapsulates all dependencies and system configurations, eliminating potential "it works on my machine" issues and simplifying deployment.

### Prerequisites

* [Python 3.9+](https://www.python.org/downloads/) (for local setup)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* An active **Microsoft Azure** subscription with the following services configured:
    * Azure OpenAI (with a model deployment)
    * Azure AI Vision
    * Azure Cosmos DB (MongoDB vCore)

### 1. Docker Deployment (Recommended)

**A. Create the Environment File**

1.  Create a file named `.env` in the root of the project directory.
2.  Add your Azure credentials to this file. This file must not be committed to version control.

    ```dotenv
    # .env file
    AZURE_OPENAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
    AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_API_VERSION="2023-05-15"
    AZURE_VISION_ENDPOINT="YOUR_AZURE_VISION_ENDPOINT"
    AZURE_VISION_KEY="YOUR_AZURE_VISION_KEY"
    COSMOS_ENDPOINT="YOUR_COSMOS_DB_MONGO_VCORE_ENDPOINT"
    COSMOS_KEY="YOUR_COSMOS_DB_PRIMARY_KEY"
    ```

**B. Build the Docker Image**

From the project's root directory (where the `Dockerfile` is located), run the following command:

```bash
docker build -t farm-app .
```

**C. Run the Docker Container**

Make sure your `.env` file is present in the root directory. Run the container using the following command, which securely passes your environment variables into the container:

```bash
docker run --env-file .env -p 8000:8000 farm-app
```

The application is now containerized and accessible at **`http://localhost:8000`**.

### 2. Local Development Setup (Alternative)

For running the application directly on your machine without Docker.

**A. Clone the Repository**

```bash
git clone <your-repository-url>
cd <repository-folder>
```

**B. Create the Environment File**
Follow the instructions in step **1.A** to create your `.env` file.

**C. Install Dependencies**

```bash
pip install -r requirements.txt
```

**D. Run the Streamlit App**

```bash
streamlit run main.py
```

The application should now be running and accessible in your web browser.
