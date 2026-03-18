# TechGear Email Support Bot 🤖

An intelligent, automated email support system for e-commerce built with **FastAPI**, **LangChain**, and a **Retrieval-Augmented Generation (RAG)** pipeline. The bot preprocesses inbound customer emails, classifies intent, routes to appropriate handlers, and generates policy-grounded email replies — or escalates to a human agent when needed.

---

## Features

- **Email Preprocessing** — Cleans signatures/disclaimers, detects language, extracts entities (customer name, order ID, SKU, VIN)
- **Intent Classification** — Zero-shot classification via `facebook/bart-large-mnli` (warranty, refund, shipping, billing, etc.)
- **Smart Routing** — Auto-routes simple queries to the RAG bot; flags complex/critical cases for human review
- **Department Detection** — Identifies the relevant product department using keyword-boosted zero-shot classification
- **RAG Engine** — Hybrid retrieval (BM25 + ChromaDB vector search) with cross-encoder reranking, powered by `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API
- **Policy-Grounded Replies** — Generates professional email replies strictly from the knowledge base (no hallucination)
- **Escalation Handling** — Gracefully escalates to human support for critical or out-of-scope queries

---

## Architecture

```
Incoming Email
     │
     ▼
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Preprocess │───▶│ Intent & Routing │───▶│  RAG Engine    │
│  (clean,    │    │  Classification  │    │  (retrieve +   │
│   NER, lang)│    │  (zero-shot LLM) │    │   rerank + LLM)│
└─────────────┘    └──────────────────┘    └────────────────┘
                            │                       │
                    Human Escalation         Email Reply
```

**RAG Pipeline detail:**
1. PDF FAQ → chunked → embedded (`BAAI/bge-large-en-v1.5`) → ChromaDB
2. BM25 (lexical) + ChromaDB (semantic) ensemble retrieval
3. Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
4. Top-3 chunks → prompt → `Qwen2.5-72B-Instruct` → formatted email reply

---

## Setup

### Prerequisites
- Python 3.10+
- A [HuggingFace](https://huggingface.co) account with an API token that has inference access

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Configuration

```bash
# Copy the example env file and fill in your credentials
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here   # Optional
HF_TOKEN=your_huggingface_token_here      # Required
PDF_PATH=/path/to/your/faqdata.pdf        # Optional, defaults to faqdata.pdf
```

> **Important:** Place your FAQ knowledge base PDF as `faqdata.pdf` in the project root (or set `PDF_PATH` in `.env`).

### Running the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API Endpoints

### `POST /preprocess`
Cleans and analyses an incoming email without generating a reply.

**Request body:**
```json
{
  "subject": "Issue with my order",
  "body": "Hi, I haven't received my order ORD-12345 yet...",
  "sender": "customer@example.com",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response:**
```json
{
  "cleaned_body": "...",
  "language": "en",
  "entities": { "order_id": "ORD-12345" },
  "intent": "shipping issue",
  "confidence": 0.87,
  "route": "general",
  "route_reason": "known issue category with sufficient confidence",
  "department": null,
  "department_confidence": null
}
```

---

### `POST /generate-reply`
Full pipeline: preprocess → route → RAG reply (or human escalation).

**Request body:** same as `/preprocess`

**Response (auto-replied):**
```json
{
  "preprocess": { ... },
  "reply": "Re: Issue with my order\n\nHi John, ...",
  "confidence": 0.91,
  "status": "success"
}
```

**Response (escalated):**
```json
{
  "preprocess": { ... },
  "reply": "Escalated to human support: potentially complex or critical content",
  "status": "escalated"
}
```

---

## Project Structure

```
.
├── main.py           # FastAPI app, email preprocessing, routing logic
├── rag_engine.py     # RAG engine: embeddings, vector store, reranker, LLM chain
├── test.py           # Basic tests
├── test_hf.py        # HuggingFace inference tests
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
└── faqdata.pdf       # Knowledge base PDF (not tracked in git)
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | ✅ Yes | HuggingFace API token for LLM inference |
| `GOOGLE_API_KEY` | ❌ Optional | Google Gemini API key (if using Gemini models) |
| `PDF_PATH` | ❌ Optional | Absolute path to FAQ PDF (defaults to `./faqdata.pdf`) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| NLP / NER | spaCy (`en_core_web_sm`) |
| Intent Classification | HuggingFace Transformers (`facebook/bart-large-mnli`) |
| Embeddings | `BAAI/bge-large-en-v1.5` via LangChain |
| Vector Store | ChromaDB |
| Lexical Search | BM25 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API |
| Orchestration | LangChain |

---

## License

MIT
