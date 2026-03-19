from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from fastapi.responses import JSONResponse
import os

from preprocessing import get_preprocessed_data

# --- Lazy RAG engine (loaded on first request) ---
_rag_engine = None
_rag_lock = asyncio.Lock()


async def get_rag_engine():
    global _rag_engine
    async with _rag_lock:
        if _rag_engine is None:
            from rag_engine import RAGEngine
            print("Initializing RAG engine...")
            try:
                engine = RAGEngine()
                await engine.initialize()
                _rag_engine = engine
            except Exception as e:
                print(f"❌ RAG Initialization failed: {str(e)}")
                raise e
        return _rag_engine


app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Email(BaseModel):
    subject: str
    body: str
    sender: str
    timestamp: str


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Running 🚀"}


@app.get("/debug/env")
def debug_env():
    """Safely check for existence of required environment variables."""
    return {
        "HF_TOKEN_SET": os.environ.get("HF_TOKEN") is not None,
        "GOOGLE_API_KEY_SET": os.environ.get("GOOGLE_API_KEY") is not None,
        "PDF_PATH_SET": os.environ.get("PDF_PATH") is not None,
        "ENV_KEYS": list(os.environ.keys())[:20]
    }


@app.post("/preprocess")
def preprocess(email: Email):
    data = get_preprocessed_data(email)
    return JSONResponse(content=data)


@app.post("/generate-reply")
async def generate_reply(email: Email):
    # 1. Preprocess
    prep = get_preprocessed_data(email)

    # 2. If route is human, don't auto-reply
    if prep["route"] == "human":
        return {
            "preprocess": prep,
            "reply": "Escalated to human support: " + prep["route_reason"],
            "status": "escalated"
        }

    # 3. RAG: Generate response
    name = prep["entities"].get("customer_name", "Valued Customer")
    engine = await get_rag_engine()
    rag_result = await engine.get_response(
        query=prep["cleaned_body"],
        customer_name=name,
        email_subject=prep["subject"]
    )

    return {
        "preprocess": prep,
        "reply": rag_result["answer"],
        "confidence": rag_result["confidence"],
        "status": "success"
    }
