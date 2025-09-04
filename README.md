
 ## 📚 Team Knowledge AI Chatbot

An AI-powered chatbot that lets your team **upload documents** and query them in natural language.
Built with **Streamlit**, **Qdrant**, **Sentence Transformers**, and **Ollama** (local LLM).



## 🚀 Features

* Upload and manage team documents (PDF, TXT, DOCX, CSV, MD).
* Ask questions and get answers with context from your documents.
* Answers are streamed from a local LLM (via Ollama).
* Smart caching:

  * **Exact cache** → instant repeat answers.
  * **Semantic cache** → reuse similar past answers.
* Admin panel with document stats and cache management.



 ## 🛠️ Prerequisites

* Python **3.9+**
* `pip` installed
* **Ollama** running locally (`ollama run llama3` to verify)
* **Qdrant** running locally (via Docker):

```bash
docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

---

 ## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/hajasaheer/team-chatbot.git
cd team-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Qdrant settings
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=   # optional if secured

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Collections
DOC_COLLECTION = os.getenv("DOC_COLLECTION", "documents")
CACHE_COLLECTION = os.getenv("CACHE_COLLECTION", "chat_cache")

# Models
DEFAULT_MODEL=llama3
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Semantic cache
SEMANTIC_CACHE_MIN_SCORE=0.5
```

---

## 📂 Project Structure

```
team-bot/
│── venv/                # Python virtual environment
│── app.py                # Streamlit entrypoint
│── pages/
│   ├── Chatbot.py        # Chat interface
│   └── Admin.py          # Document ingestion & cache management
│── utils/
│   ├── qa.py             # Qdrant, embeddings, caching, Ollama logic
│   ├── ingestion.py      # PDF/text extraction, chunking, embedding
│   └── ollama_client.py  # (optional Ollama helpers)
│── .env                  # Configuration
```

---

## ▶️ Running the Application

Start Streamlit manually:

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Or set up as a **systemd service**:

```ini
[Unit]
Description=Team Bot - Streamlit LangChain Chatbot
After=network.target

[Service]
User=root
WorkingDirectory=/root/team-bot
ExecStart=/root/team-bot/venv/bin/streamlit run /root/team-bot/app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 💡 Usage

### Main Menu

* Navigate between **Chat** and **Admin** pages.

### Chatbot Page

1. Enter a question.
2. Bot will:

   * Check cache for past answers.
   * Retrieve context from Qdrant.
   * Build a prompt.
   * Stream an answer from Ollama.
   * Cache the result.

### Admin Page

* Upload new documents for ingestion.
* View Qdrant debug info & collection stats.
* Clear or delete cache entries.

---

## ⚙️ How It Works

1. **Document Ingestion**

   * PDFs parsed with PyPDF2.
   * Text chunked (500 chars, 50 overlap).
   * Chunks embedded with SentenceTransformers.
   * Stored in Qdrant with metadata.

2. **Question Answering**

   * Query → embedded and searched in Qdrant.
   * Top matches → context for the LLM.
   * Prompt sent to Ollama model.
   * Answer streamed back and cached.

3. **Cache**

   * **Exact cache** → instant match.
   * **Semantic cache** → similarity match above threshold.

---

## 🔄 Example Flow

1. Admin → Upload `manual.pdf`.
2. Chat → Ask: *“What are the safety procedures?”*
3. Bot retrieves relevant chunks and responds with citations.
4. Repeat same query → instant cached response.

---

👉 Project Repository: [github.com/hajasaheer/team-chatbot](https://github.com/hajasaheer/team-chatbot)

---

