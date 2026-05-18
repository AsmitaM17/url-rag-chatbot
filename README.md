URL RAG Chatbot
This project is a single-page Streamlit chatbot that reads one URL at a time, chunks the web page in memory, stores those chunks in an in-memory ChromaDB vector store, and answers questions with Ollama running gemma2:2b.

What You Are Building
UI: Streamlit
Vector database: ChromaDB ephemeral memory mode
Embeddings: sentence-transformers/all-MiniLM-L6-v2
LLM engine: Ollama with gemma2:2b
Input: one web URL
Storage: temporary memory only
When you close the app, the vector store disappears.

End-to-End Flow
You paste a URL.
Python downloads the page HTML.
BeautifulSoup extracts readable text.
The text is split into overlapping chunks.
MiniLM converts each chunk into an embedding vector.
ChromaDB stores those vectors in memory.
You ask a question.
MiniLM embeds your question.
ChromaDB finds the most relevant chunks.
Ollama gemma2:2b receives the question plus retrieved chunks.
Streamlit displays the answer.
Install Ollama
Download and install Ollama from:

https://ollama.com/download

Then pull the model:

ollama pull gemma2:2b
Keep Ollama running in the background.

Create a Python Environment
From this project folder:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
The first install can take a few minutes because sentence-transformers, torch, and chromadb are larger packages.

Run the App
streamlit run app.py
Open the local URL Streamlit prints, usually:

http://localhost:8501
Good URLs to Test
Wikipedia articles
Blog posts
Documentation pages
News articles that are not behind a paywall
Some websites block automated reading. If that happens, try another URL.

Why This Fits an 8 GB RAM Laptop
The app only processes one URL at a time. It does not manage a permanent document database, does not store PDFs, and does not keep old files around. ChromaDB runs in ephemeral memory mode, and gemma2:2b is small enough to run locally on modest machines through Ollama.

Project Files
app.py: complete Streamlit chatbot
requirements.txt: Python dependencies
README.md: setup and learning guide