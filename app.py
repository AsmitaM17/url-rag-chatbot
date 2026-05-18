import hashlib
import re
from typing import Dict, List, Tuple

import chromadb
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


st.set_page_config(
    page_title="URL RAG Chatbot",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading MiniLM embedding model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()

    title = clean_text(soup.title.get_text(" ")) if soup.title else url

    article = soup.find("article")
    source = article if article else soup.body or soup
    paragraphs = [clean_text(p.get_text(" ")) for p in source.find_all(["p", "h1", "h2", "h3", "li"])]
    text = "\n".join(p for p in paragraphs if len(p) > 40)

    if not text:
        text = clean_text(source.get_text(" "))

    if len(text) < 300:
        raise ValueError("Could not extract enough readable text from this URL.")

    return title, text


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


def create_memory_vector_store(url: str, chunks: List[str]) -> chromadb.Collection:
    client = chromadb.EphemeralClient()
    collection_name = "url_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    collection = client.get_or_create_collection(collection_name)

    model = load_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

    collection.add(
        ids=[f"chunk-{idx}" for idx in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": url, "chunk": idx} for idx in range(len(chunks))],
    )
    return collection


def retrieve_context(collection: chromadb.Collection, question: str) -> List[str]:
    model = load_embedding_model()
    question_embedding = model.encode([question], normalize_embeddings=True).tolist()[0]
    result = collection.query(query_embeddings=[question_embedding], n_results=TOP_K)
    return result.get("documents", [[]])[0]


def build_prompt(question: str, context_chunks: List[str], title: str, url: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return f"""You are a helpful chatbot answering questions about one web page.

Use only the context below. If the answer is not in the context, say you do not know from this page.
Keep the answer clear and concise.

Page title: {title}
URL: {url}

Context:
{context}

Question: {question}
Answer:"""


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
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


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

st.title("URL RAG Chatbot")
st.caption("Ask questions about one live web page using Streamlit, ChromaDB, MiniLM, and Ollama Gemma 2B.")

with st.sidebar:
    st.header("Setup")
    url = st.text_input(
        "Web page URL",
        placeholder="https://en.wikipedia.org/wiki/Streamlit",
    )
    load_clicked = st.button("Load URL", type="primary", use_container_width=True)
    clear_clicked = st.button("Clear memory", use_container_width=True)

    st.divider()
    st.write("Models")
    st.code(f"Embedding: {EMBEDDING_MODEL}\nLLM: {OLLAMA_MODEL}", language="text")

if clear_clicked:
    reset_chat()
    st.rerun()

if load_clicked:
    if not url.startswith(("http://", "https://")):
        st.sidebar.error("Please enter a full URL starting with http:// or https://")
    else:
        try:
            with st.spinner("Reading the page and building an in-memory vector store..."):
                title, text = fetch_url_text(url)
                chunks = chunk_text(text)
                collection = create_memory_vector_store(url, chunks)
                st.session_state.collection = collection
                st.session_state.page = {
                    "title": title,
                    "url": url,
                    "characters": len(text),
                    "chunks": len(chunks),
                }
                st.session_state.loaded_url = url
                st.session_state.messages = []
        except Exception as exc:
            st.sidebar.error(f"Could not load page: {exc}")

page = st.session_state.page
if page:
    st.success(
        f"Loaded: {page['title']} | {page['characters']:,} characters | {page['chunks']} chunks"
    )
else:
    st.info("Paste a URL in the sidebar, load it, then ask questions about that page.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about the loaded page...")

if question:
    if st.session_state.collection is None or st.session_state.page is None:
        st.warning("Load a URL first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the page context and asking Ollama..."):
                context_chunks = retrieve_context(st.session_state.collection, question)
                prompt = build_prompt(
                    question=question,
                    context_chunks=context_chunks,
                    title=st.session_state.page["title"],
                    url=st.session_state.page["url"],
                )
                answer = ask_ollama(prompt)
                st.markdown(answer)

                with st.expander("Retrieved context"):
                    for idx, chunk in enumerate(context_chunks, start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.write(chunk)

        except requests.exceptions.ConnectionError:
            answer = (
                "I could not connect to Ollama. Start Ollama and run "
                f"`ollama pull {OLLAMA_MODEL}` first."
            )
            st.error(answer)
        except Exception as exc:
            answer = f"Something went wrong: {exc}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})