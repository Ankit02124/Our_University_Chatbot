"""Tests that run without an API key: cleaning, chunking and the answer step
with a fake embedding model and a fake chat model."""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.vectorstores import InMemoryVectorStore

import rag
from clean_data import clean_text


def test_clean_text_removes_tags_and_page_numbers():
    raw = "<page 3>\nAttendance\n\n\n\n12\nStudents must attend 75% of classes.\n"
    assert clean_text(raw) == "Attendance\n\nStudents must attend 75% of classes."


def test_chunks_respect_size_and_keep_source(tmp_path):
    words = " ".join(f"rule{i}" for i in range(2000))
    (tmp_path / "policy.txt").write_text(words, encoding="utf-8")
    chunks = rag.load_chunks(tmp_path)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= rag.CHUNK_SIZE for c in chunks)
    assert all(c.metadata["source"].endswith("policy.txt") for c in chunks)
    # neighbouring chunks overlap
    assert chunks[0].page_content[-50:] in chunks[1].page_content


def test_doc_title_handles_windows_paths():
    doc = Document("x", metadata={"source": "data_clean\\Mop_UG_AY 2023-24.txt"})
    assert rag.doc_title(doc) == rag.DOC_TITLES["Mop_UG_AY 2023-24.txt"]
    assert rag.doc_title(Document("x", metadata={"source": "data_clean/other.txt"})) == "other.txt"


def test_answer_uses_retrieved_chunks():
    attendance = "Students are required to attend a minimum of 75% of classes in all courses."
    hostel = "The hostel gates are closed from 10:00 PM to 05:00 AM every day."
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=64))
    store.add_documents([Document(attendance), Document(hostel)])
    retriever = store.as_retriever(search_kwargs={"k": 1})
    llm = FakeListChatModel(responses=["75% attendance is required."])

    text, docs = rag.answer(attendance, retriever, llm)

    assert text == "75% attendance is required."
    assert [d.page_content for d in docs] == [attendance]


def test_prompt_contains_context_and_fallback():
    msg = rag.PROMPT.format(context="CONTEXT", question="QUESTION")
    assert "CONTEXT" in msg and "QUESTION" in msg and rag.NOT_FOUND in msg
