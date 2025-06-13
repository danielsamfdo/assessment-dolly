# Overview

This application provides an interactive chat experience with "Dolly" — a digital twin constructed from textual and PDF-based documents. Using LangChain, OpenAI, and vector search (FAISS or Pinecone), users can ask Dolly contextual questions about life, legacy, and more. Built with Streamlit, the app offers a conversational UI interface.

🧱 Dependencies
streamlit: For building the UI.

pypdf: For reading PDF content.

langchain: For document parsing, embedding, and conversational chains.

pinecone: For cloud-based vector storage.

dotenv: For loading API keys from .env.


# Python Virtual Env

In the downloaded directory do the below

`python -m venv venv`


`source venv/bin/activate`


# Run the Installer for requirements.txt

`python3 -m pip install -r requirements.txt`


# Add your OPENAI API Key 

export OPENAI_API_KEY="your_openai_api_key"

# Add your PINECONE_API_KEY if u want to use pinecone as your vector DB.

export PINECONE_API_KEY="api_key"


# To Run the APP
`streamlit run app.py --logger.level=error`


# Documentation:

* https://platform.openai.com/docs/guides/batch Batch API for reducing costs for content embeddings

* Pinecone for Embeddings to work across various Client Interfaces [Eg. Mobile, Web, ] and support for various models and various Cloud Providers like GCP etc

* https://docs.pinecone.io/guides/get-started/overview 



3 stages

1. Data Process and Indexing
2. Store in DB [vector -> text chunks]
3. Identify closest pairs to the query and pass that as context to the prompt.