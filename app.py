import hashlib
import re
from typing import Dict, List, Tuple

import chromadb
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="URL RAG Chatbot",
    layout="wide",
)


# =========================
# SESSION STATE
# =========================

def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.collection = None
    st.session_state.page = None
    st.session_state.loaded_url = ""


def initialize_state() -> None:
    defaults: Dict[str, object] = {
        "messages": [],
        "collection": None,
        "page": None,
        "loaded_url": "",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialize_state()


# =========================
# LOAD EMBEDDING MODEL
# =========================

@st.cache_resource(show_spinner="Loading MiniLM embedding model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# FETCH WEBPAGE
# =========================

def fetch_url_text(url: str) -> Tuple[str, str]:

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "header",
        "footer",
        "nav"
    ]):
        tag.decompose()

    title = clean_text(soup.title.get_text(" ")) if soup.title else url

    article = soup.find("article")
    source = article if article else soup.body or soup

    paragraphs = [
        clean_text(p.get_text(" "))
        for p in source.find_all(["p", "h1", "h2", "h3", "li"])
    ]

    text = "\n".join(
        p for p in paragraphs if len(p) > 40
    )

    if not text:
        text = clean_text(source.get_text(" "))

    if len(text) < 300:
        raise ValueError(
            "Could not extract enough readable text from this URL."
        )

    return title, text


# =========================
# CHUNK TEXT
# =========================

def chunk_text(text: str) -> List[str]:

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# =========================
# CREATE VECTOR STORE
# =========================

def create_memory_vector_store(
    url: str,
    chunks: List[str]
) -> chromadb.Collection:

    client = chromadb.EphemeralClient()

    collection_name = (
        "url_" +
        hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    )

    collection = client.get_or_create_collection(
        collection_name
    )

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[f"chunk-{idx}" for idx in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": url,
                "chunk": idx
            }
            for idx in range(len(chunks))
        ],
    )

    return collection


# =========================
# RETRIEVE CONTEXT
# =========================

def retrieve_context(
    collection: chromadb.Collection,
    question: str
) -> List[str]:

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
    )

    return result.get("documents", [[]])[0]


# =========================
# BUILD PROMPT
# =========================

def build_prompt(
    question: str,
    context_chunks: List[str],
    title: str,
    url: str
) -> str:

    context = "\n\n---\n\n".join(context_chunks)

    return f"""
You are a helpful chatbot answering questions about one web page.

Use ONLY the context below.

If the answer is not in the context,
say you do not know from this page.

Keep answers concise and factual.

Page title:
{title}

URL:
{url}

Context:
{context}

Question:
{question}

Answer:
"""


# =========================
# ASK OLLAMA
# =========================

def ask_ollama(prompt: str) -> str:

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "").strip()


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>

    :root {
        --bg: #fbfaf7;
        --panel: #ffffff;
        --ink: #2f3437;
        --muted: #6d7378;
        --line: #ece7df;
        --accent: #6f8f72;
        --accent-soft: #edf4ec;
    }

    .stApp {
        background:
            radial-gradient(circle at top left,
            rgba(255, 241, 238, 0.85),
            transparent 32rem),
            linear-gradient(180deg,
            var(--bg),
            #f7f4ef);

        color: var(--ink);
    }

    .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 750 !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid #ececec;
        padding: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# UI
# =========================

st.title("🌐 URL RAG Chatbot")

st.caption(
    "Chat with any webpage using Streamlit, "
    "ChromaDB, MiniLM embeddings, and Ollama Gemma 2B."
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("Setup")

    url = st.text_input(
        "Enter webpage URL",
        placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
    )

    load_clicked = st.button(
        "Load URL",
        type="primary",
        use_container_width=True,
    )

    clear_clicked = st.button(
        "Clear Memory",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Models")

    st.code(
        f"Embedding: {EMBEDDING_MODEL}\n"
        f"LLM: {OLLAMA_MODEL}",
        language="text",
    )


# =========================
# CLEAR MEMORY
# =========================

if clear_clicked:
    reset_chat()
    st.rerun()


# =========================
# LOAD URL
# =========================

if load_clicked:

    if not url.startswith(("http://", "https://")):

        st.sidebar.error(
            "Please enter a valid URL starting with http:// or https://"
        )

    else:

        try:

            with st.spinner(
                "Reading webpage and building vector store..."
            ):

                title, text = fetch_url_text(url)

                chunks = chunk_text(text)

                collection = create_memory_vector_store(
                    url,
                    chunks,
                )

                st.session_state.collection = collection

                st.session_state.page = {
                    "title": title,
                    "url": url,
                    "characters": len(text),
                    "chunks": len(chunks),
                }

                st.session_state.loaded_url = url

                st.session_state.messages = []

            st.sidebar.success("Page loaded successfully!")

        except Exception as exc:

            st.sidebar.error(
                f"Could not load webpage:\n\n{exc}"
            )


# =========================
# PAGE STATUS
# =========================

page = st.session_state.page

if page:

    st.success(
        f"Loaded: {page['title']} | "
        f"{page['characters']:,} characters | "
        f"{page['chunks']} chunks"
    )

else:

    st.info(
        "Load a webpage URL from the sidebar to begin chatting."
    )


# =========================
# CHAT HISTORY
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================
# CHAT INPUT
# =========================

question = st.chat_input(
    "Ask something about the webpage..."
)


# =========================
# HANDLE QUESTIONS
# =========================

if question:

    if (
        st.session_state.collection is None
        or
        st.session_state.page is None
    ):

        st.warning("Please load a webpage first.")

        st.stop()

    # USER MESSAGE

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ASSISTANT MESSAGE

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching context and asking Gemma..."
            ):

                context_chunks = retrieve_context(
                    st.session_state.collection,
                    question,
                )

                prompt = build_prompt(
                    question=question,
                    context_chunks=context_chunks,
                    title=st.session_state.page["title"],
                    url=st.session_state.page["url"],
                )

                answer = ask_ollama(prompt)

                st.markdown(answer)

                with st.expander("Retrieved Context"):

                    for idx, chunk in enumerate(
                        context_chunks,
                        start=1
                    ):

                        st.markdown(f"**Chunk {idx}**")

                        st.write(chunk)

        except requests.exceptions.ConnectionError:

            answer = (
                "Could not connect to Ollama.\n\n"
                "Start Ollama first and run:\n\n"
                f"`ollama pull {OLLAMA_MODEL}`"
            )

            st.error(answer)

        except Exception as exc:

            answer = f"Something went wrong:\n\n{exc}"

            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )