"""Document loading, the Chroma index and the answer step for the chatbot.

Kept separate from app.py so it can be tested without Streamlit or an API key.
"""
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_clean"
DB_PATH = ROOT / "vector_db_mistral"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

EMBED_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-large-latest"

NOT_FOUND = "I cannot find that information in the official documents."

# readable names for the files in data_clean/
DOC_TITLES = {
    "Mop_UG_AY 2023-24.txt": "Student Manual of Policies, UG programmes (2023)",
    "345_Resident_Student_Policy_2025_26_NC.txt": "Resident Student (Hostel) Policy 2025-26",
    "MOP1.txt": "Student Manual of Policies, Mass Communication (2020)",
}

PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant for XIM University students.\n"
    "Answer the question using only the context below, which is taken from the "
    "university's official policy documents. Keep the answer short and quote "
    "numbers (percentages, times, fees) exactly as they appear.\n"
    f'If the answer is not in the context, reply: "{NOT_FOUND}"\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)


def load_chunks(data_path=DATA_PATH):
    """Read every .txt file in data_path and split it into overlapping chunks."""
    docs = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
        for path in sorted(Path(data_path).glob("*.txt"))
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


def get_vector_db(embeddings, db_path=DB_PATH, data_path=DATA_PATH):
    """Open the saved Chroma index, or build it from the documents the first time."""
    if Path(db_path).exists():
        return Chroma(persist_directory=str(db_path), embedding_function=embeddings)
    return Chroma.from_documents(load_chunks(data_path), embeddings, persist_directory=str(db_path))


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def doc_title(doc):
    # the saved index was built on Windows, so the path may use backslashes
    name = re.split(r"[\\/]", doc.metadata.get("source", ""))[-1]
    return DOC_TITLES.get(name, name or "unknown document")


def answer(question, retriever, llm):
    """Return the model's answer and the chunks it was given."""
    docs = retriever.invoke(question)
    chain = PROMPT | llm | StrOutputParser()
    text = chain.invoke({"context": format_docs(docs), "question": question})
    return text, docs
