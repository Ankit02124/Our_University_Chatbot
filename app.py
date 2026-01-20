import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
DATA_PATH = "data_clean"
DB_PATH = "vector_db_mistral"

# --- PAGE SETUP ---
st.set_page_config(page_title="University Bot", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.caption("Powered by Mistral AI • Capstone Project")

# --- SIDEBAR: API KEY ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Mistral API Key:", type="password")
    if not api_key:
        st.warning("Please enter your Mistral API key to start.")
        st.stop()
    os.environ["MISTRAL_API_KEY"] = api_key
    st.success("API Key Accepted!")

# --- LOAD DATA ---
@st.cache_resource
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Data not found. Please run clean_data.py first.")
        st.stop()
    
    with st.spinner("Processing University Manuals..."):
        # Load text files
        loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        docs = loader.load()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Create Vector Database
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
        return vector_db

# Check if DB exists, if not create it
if os.path.exists(DB_PATH):
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
else:
    vector_db = load_data()

# --- RAG ENGINE ---
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatMistralAI(model="mistral-large-latest", temperature=0.1)

# System Prompt
template = """
You are a helpful AI assistant for the University. 
Answer questions strictly based on the provided context. 
If the answer is not in the context, say "I cannot find that information in the official documents."

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I can answer questions about Academic Rules, Hostel Policy, and Mass Comm guidelines. What would you like to know?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(user_input)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})