# dolly_app.py
from pypdf import PdfReader
import streamlit as st
# from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Any

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

import os
import dotenv


# Load environment variables
dotenv.load_dotenv()

def get_openai_api_key():
    """
    Get OpenAI API key from environment variable.
    Returns the API key as a string.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    return api_key

def get_pinecone_api_key():
    """
    Get Pinecone API key from environment variable.
    Returns the API key as a string.

    Only necessary for experimental running. When using Pinecone yourself, 
    you can use environment variables.
    """
    api_key = os.environ.get("PINECONE_API_KEY")
    return api_key

PINECONE_API_KEY = get_pinecone_api_key()
OPENAI_API_KEY = get_openai_api_key()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_local_llm():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # or any other causal LM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=pipe)


# --- Load Dolly Docs ---
def load_dolly_documents():
    base_path = "/Users/tn_family/Documents/GitHub/assessment-dolly/data"
    doc_list = []
    idx = 1
    for file_name in os.listdir(base_path):
        fpath = os.path.join(base_path, file_name)
        if file_name.endswith(".txt"):
            try:
                # Read the TXT file
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # print(content)
                    doc_list.append(Document(page_content=content, metadata={"source": file_name, "id": idx}))
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        elif file_name.endswith(".pdf"):
            try:
                # Read the PDF
                content = ""
                reader = PdfReader(fpath)
                for i in range(len(reader.pages)):
                    content += reader.pages[i].extract_text()
                doc_list.append(Document(page_content=content, metadata={"source": file_name, "id": idx}))
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        idx += 1
    return doc_list

# --- Prepare Vector DB ---
def build_vectorstore_with_MiniLM(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)  # `documents` must be List[Document]
    # Build FAISS vectorstore
    vectordb = FAISS.from_documents(texts, embedding=embedding_model)
    return vectordb

# --- Prepare Vector DB ---
def build_vectorstore_with_OpenAI(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(texts, embedding=embeddings)
    return vectordb



def generate_ids(doc_chunk, idx):
    id_num = doc_chunk.metadata['id']
    src = doc_chunk.metadata['source'] if 'source' in doc_chunk.metadata else "na"
    return f"release_{id_num}#feature_{src}#chunk_num{idx}"

def build_vectorstore_via_pinecone(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    # PINECONE CONFIGURATION
    index_name = "dolly-assessment"
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
    )
    if not pc.has_index(index_name):
        # Create the INDEX
        pc.create_index(
            name=index_name,
            # dimension of the vector embeddings produced by OpenAI's text-embedding-3-small
            dimension=1536,
            metric="cosine",
            # parameters for the free tier index
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Initialize index client
        index = pc.Index(name=index_name)
        # View index stats
        index.describe_index_stats()
    else:
        index = pc.Index(name=index_name)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    ids = [generate_ids(doc, idx) for idx, doc in enumerate(documents)]
    # If you want to store this on the cloud, you need to invoke upsert.
    # TODO(invoke Upsert)
    vector_store.add_documents(documents=documents, ids=ids)
    return vector_store


# --- Set up QA Chain with Memory ---
def setup_conversational_retrieval_chain_with_openAI(vectorstore: PineconeVectorStore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # https://platform.openai.com/settings/organization/limits
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4.1-nano", temperature=0.7)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)
    return qa_chain

def setup_conversation_with_model(retriever, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    

    # Prompt
    prompt = PromptTemplate.from_template("""
    You are a helpful digital twin of Dolly. Use the following conversation history and retrieved documents to answer the question.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}
    """)

    # Chain components
    chain = (
        {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(lambda q: retriever.get_relevant_documents(q)),
            "chat_history": RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, memory


def converse(query: str, memory: ConversationBufferMemory, chain: Any):
    response = chain.invoke(query)
    memory.save_context({"input": query}, {"output": response})


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Talk to Dolly 🐑", layout="centered")
    st.title("🐑 Talk to Dolly — Your Woolly Digital Twin")
    st.markdown("Ask Dolly anything about her life, her thoughts, her family, or her legacy.")

    if "qa" not in st.session_state:
        with st.spinner("Loading Dolly's memories..."):
            docs = load_dolly_documents()
            st.session_state.vectordb = build_vectorstore_with_MiniLM(docs)
            # chain, memory = setup_conversation_with_model(st.session_state.vectordb.as_retriever(), llm = ChatOpenAI(model="o4-mini", openai_api_key=OPENAI_API_KEY))
            chain, memory = setup_conversation_with_model(st.session_state.vectordb.as_retriever(), load_local_llm())
            st.session_state.mem = memory 
            st.session_state.chain = chain

    user_input = st.text_input("You:", placeholder="Hey Dolly, what do you think of cloning ?")
    if user_input:
        response = converse(user_input, st.session_state.mem, st.session_state.chain)
        st.markdown(f"**Dolly:** {response}")

if __name__ == "__main__":
    main()
