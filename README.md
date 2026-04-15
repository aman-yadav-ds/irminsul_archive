# 🌍 Teyvat Lore Graph
### *A knowledge engine built for the world of Genshin Impact — where a graph database meets a vector store to answer every lore question you've ever had.*

Note :- Don't try to be smart thinking you could stole that gemini key and use it. I already revoked it the moment it was pushed.

---

## The Story Behind It

Genshin Impact doesn't just have a game — it has a *world*. A world with seven nations, dozens of playable characters, ancient gods, secret histories, and an ocean of lore scattered across item descriptions, character stories, quest dialogues, and archon quests. For a fan who wants to understand how everything connects — who Zhongli was before the war, why the Abyss Order exists, what the Sages of Sumeru are hiding — the answer is rarely a single search result. It's a *web* of relationships.

This project was born from that exact frustration. **Teyvat Lore Graph** is a personal knowledge system built from scratch — a hybrid AI pipeline that treats the lore of Teyvat not as a flat document, but as a living, interconnected graph. You ask a question in plain English. The system reasons across a Neo4j knowledge graph and a ChromaDB vector store simultaneously, then returns you a contextual answer *and* a live visualization of the relationships it found.

It is part library, part reasoning engine, part love letter to a game that takes its world-building very seriously.

---

## What It Does

At its core, Teyvat Lore Graph is a **Hybrid RAG (Retrieval-Augmented Generation)** system with a web interface. Here's what happens when you ask it something:

1. Your question arrives at a Flask API endpoint.
2. The `LoreReasoner` engine wakes up and runs a dual-retrieval strategy — fetching semantically similar lore passages from **ChromaDB** and traversing structured relationships in **Neo4j**.
3. The retrieved context is fed into an LLM (powered by LangChain, with support for Google Gemini, Groq-hosted models, and local Ollama models).
4. The LLM generates a grounded, lore-accurate answer.
5. Simultaneously, the knowledge graph edges that were traversed are extracted and sent back as structured node-edge data.
6. The frontend renders that data as an **interactive graph visualization** using Vis.js, so you can see *exactly* which entities and relationships the answer was built from.

It's not just Q&A — it's *explainable* Q&A.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Flask Web App                     │
│              (app.py  /  templates/)                │
└───────────────────────┬─────────────────────────────┘
                        │  POST /api/ask
                        ▼
┌─────────────────────────────────────────────────────┐
│               LoreReasoner (RAG Engine)             │
│            src/pipeline/rag_engine.py               │
│                                                     │
│   ┌─────────────────┐    ┌────────────────────────┐ │
│   │  Vector Search  │    │   Graph Traversal      │ │
│   │   (ChromaDB)    │    │      (Neo4j)           │ │
│   │                 │    │                        │ │
│   │ Sentence-       │    │ LangChain Neo4j        │ │
│   │ Transformers    │    │ Cypher Chain           │ │
│   │ Embeddings      │    │                        │ │
│   └────────┬────────┘    └────────────┬───────────┘ │
│            │                         │             │
│            └──────────┬──────────────┘             │
│                       ▼                             │
│              Combined Context Window                │
│                       │                             │
│                       ▼                             │
│         LLM (Gemini / Groq / Ollama)               │
│              via LangChain                          │
└───────────────────────┬─────────────────────────────┘
                        │
              ┌─────────┴──────────┐
              │  answer (text)     │
              │  graph_edges (JSON)│
              └─────────┬──────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│               Frontend (Vis.js)                     │
│   Interactive graph + text answer rendered live     │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Web Framework** | Flask | API server and HTML template rendering |
| **Graph Database** | Neo4j | Stores structured lore relationships (entities, factions, events) |
| **Vector Database** | ChromaDB | Stores lore passages as semantic embeddings for similarity search |
| **Embeddings** | Sentence-Transformers | Converts text into dense vectors for ChromaDB retrieval |
| **LLM Orchestration** | LangChain | Chains together retrieval, context injection, and generation |
| **LLM Backends** | Google Gemini / Groq / Ollama | Interchangeable generation backends |
| **Graph Visualization** | Vis.js | Interactive node-edge graph rendered in the browser |
| **Data Ingestion** | BeautifulSoup4 + Requests | Scraping and parsing lore content |
| **Package Management** | uv | Fast Python dependency management |
| **Python Version** | 3.12 | |

---

## Project Structure

```
teyvat_lore_graph/
│
├── app.py                  # Flask app — API routes and Vis.js payload formatting
│
├── src/
│   └── pipeline/
│       └── rag_engine.py   # LoreReasoner — the heart of the hybrid RAG system
│       └── extractor.py    # LoreExtractor — Automated Populating the GraphDB with Knowledge
│       └── scraper.py      # GenshinSmartScrapper — Automated Scrapping of Data from the Genshin Wiki
│       └── vector_db_build.py # VectorDBBuilder — Populated the vector DB with Scrapped Knowledge
│
├── static/                 # CSS and JavaScript (including Vis.js graph rendering)
├── templates/              # HTML templates (Jinja2)
│
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Locked dependency tree
└── .python-version         # Pinned to Python 3.12
```

---

## Getting Started

### Prerequisites

Make sure you have the following running before you start:

- **Neo4j** — a local or cloud instance (Neo4j Desktop or AuraDB both work)
- **Python 3.12** — exactly, as pinned in `.python-version`
- **uv** — the package manager used in this project (`pip install uv`)
- An API key for at least one LLM provider: **Google Gemini** or **Groq**. Alternatively, have **Ollama** installed and running locally.

### Installation

```bash
# Clone the repository
git clone https://github.com/aman-yadav-ds/teyvat_lore_graph.git
cd teyvat_lore_graph

# Install dependencies using uv
uv sync
```

### Environment Variables

Create a `.env` file in the project root and fill in your credentials:

```env
# Neo4j
NEO4J_URI=neo4j+ssc://**************.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# LLM Provider (choose one or more)
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

# Ollama (if running locally, no key needed — just set the model)
OLLAMA_MODEL=llama3
```

A simple note if you are querying you database in production use URI as `neo4j+s` if you are just running it locally use URI `neo4j+ssc` as I have used.

### Running the App

```bash
# Start the Flask development server
python app.py
```

Navigate to `http://localhost:5000` in your browser. You'll be greeted with the lore interface — type any question about Teyvat and watch the graph answer back.

---

## How the Hybrid RAG Works

The design philosophy here is that no single retrieval method is complete on its own.

**Vector search alone** is great at finding *thematically similar* passages, but it has no concept of structure. It can't tell you that Zhongli *is* the Geo Archon, or that the Fatui *serves under* the Tsaritsa — it can only surface documents that *talk about* those things.

**Graph traversal alone** is precise and structured, but it only knows what's been explicitly modeled. It can't handle vague, open-ended questions, paraphrases, or queries that don't map cleanly to a Cypher query.

**Together**, they cover each other's blind spots. ChromaDB catches the broad semantic intent of a query, while Neo4j provides the hard relational facts. The LLM then synthesizes both streams into a single coherent answer, grounded in actual data.

This is what makes the system feel less like a search engine and more like a scholar who has read everything and remembers how it all connects.

---

## Skills Demonstrated

This project was built as a portfolio piece to showcase a practical command of modern AI/data engineering concepts:

- **Graph database design** — modeling a rich, interconnected domain (characters, nations, factions, archons, events) as a property graph in Neo4j
- **Vector database integration** — ingesting, embedding, and querying lore text using ChromaDB and Sentence-Transformers
- **Hybrid RAG architecture** — combining structured (graph) and unstructured (vector) retrieval into a single reasoning pipeline
- **LangChain orchestration** — building multi-component LLM chains with swappable backends (Gemini, Groq, Ollama)
- **Full-stack development** — Flask API, Jinja2 templating, vanilla JS + CSS frontend
- **Interactive data visualization** — rendering live knowledge graph results using Vis.js
- **Clean Python packaging** — using `uv`, `pyproject.toml`, and `.python-version` for reproducible environments

---

## A Note on the Data

The lore data that populates the Neo4j graph and ChromaDB store comes from publicly available Genshin Impact lore sources — item descriptions, character backstories, and in-game books. The scraping pipeline (built with BeautifulSoup4 and Requests) processes and structures this raw text into the graph schema before ingestion.

*This project is a fan-made, non-commercial tool. Genshin Impact and all related content belong to HoYoverse.*

---

## What's Next

A few directions this project could grow into:

- **Automated graph updates** — scraping new lore as game patches drop and updating Neo4j incrementally
- **Multi-hop graph reasoning** — following chains of relationships deeper (e.g., "What connects Kaeya to the Abyss Order?")
- **Timeline-aware queries** — treating in-game history as a temporal graph, so questions about past events have proper chronological context
- **Dockerized deployment** — packaging the full stack (Flask + Neo4j + ChromaDB) into a `docker-compose` setup for easy one-command startup

#### But before any of that there is some descripencies when i populated the neo4j database and here's that picture:
Since i used the qwen2.5-coder:7b model running locally, the relationships between charactors, faction and events isn't that of a good quality there might be sometimes when a realtionship is reversed. This is mainly because i didn't have time to physically check every relationship the model was extracting and I refined the Extractor as much as i could. I thought of a supervisor-worker type architecture because i was also working in an autonomous agent and that would make the supervisor to make changes in the relationship it extracted and then either correct them or make the worker redo it pointing that mistake out.

I also didn't have the hardware to use a more reasonable model and the money to use cloud services. So if any of you who is reading this or is interested in this project want to Refine the Extractor or have some idea so the GraphDB is populated with data which is 99.9% accurate. We can work on this together.


---

## Running Gemini or Groq Models

If you are thinking of using cloud based services then just change the model definition in the `__init__` method to the below

1. First Install Langchain with google support

* Gemini
```bash
uv add langchain-google-genai
```

* Groq
```bash
uv add langchain-groq
```

2. Setup Environment Variable
```
GROQ_API_KEY=*************************
GOOGLE_API_KEY=************************
```

3. Change Model Definition in `rag_engine.py`

* Gemini
```python
from langchain_google_genai import ChatGoogleGenerativeAI
self.llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params... 
)
```

* Groq
```python
from langchain_groq import ChatGroq
self.llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)
```

Note :- I have used three prompts and three model definition for three steps. You can choose which step you want to change the model for.


*Built by [Amandeep Yadav](https://github.com/aman-yadav-ds) — because the lore deserves a proper library.*