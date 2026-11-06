import os

import streamlit as st
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

import rag

st.set_page_config(page_title="XIMI - XIM University Chatbot", page_icon="🎓")
st.title("🎓 XIMI - XIM University Chatbot")
st.caption("Answers come from the university's official policy documents.")

GREETING = (
    "Hello! Ask me about academic rules, attendance, exams and grading, "
    "scholarships or hostel rules."
)

with st.sidebar:
    st.header("Settings")
    api_key = os.environ.get("MISTRAL_API_KEY") or st.text_input("Mistral API key", type="password")
    st.subheader("Documents")
    for title in rag.DOC_TITLES.values():
        st.markdown(f"- {title}")
    if st.button("Clear chat"):
        st.session_state.messages = []

if not api_key:
    st.info("Enter your Mistral API key in the sidebar to start.")
    st.stop()


@st.cache_resource(show_spinner="Loading the policy documents...")
def load_retriever(key):
    embeddings = MistralAIEmbeddings(model=rag.EMBED_MODEL, api_key=key)
    db = rag.get_vector_db(embeddings)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": rag.TOP_K})


@st.cache_resource
def load_llm(key):
    return ChatMistralAI(model=rag.CHAT_MODEL, temperature=0.1, api_key=key)


def show_sources(docs):
    with st.expander("Sources"):
        for doc in docs:
            st.markdown(f"**{rag.doc_title(doc)}**")
            st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))


retriever = load_retriever(api_key)
llm = load_llm(api_key)

if not st.session_state.get("messages"):
    st.session_state.messages = [{"role": "assistant", "content": GREETING, "sources": []}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

if question := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                text, docs = rag.answer(question, retriever, llm)
        except Exception as err:  # bad key, rate limit, network
            st.error(f"Mistral API request failed: {err}")
            st.stop()
        st.write(text)
        show_sources(docs)
    st.session_state.messages.append({"role": "assistant", "content": text, "sources": docs})
