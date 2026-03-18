from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re, spacy, langdetect, asyncio
from transformers import pipeline
from rag_engine import RAGEngine
from fastapi.responses import JSONResponse


classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp=spacy.load("en_core_web_sm")

# Initialize RAG Engine
rag_engine = RAGEngine()

@app.on_event("startup")
async def startup_event():
    await rag_engine.initialize()
    print("🚀 RAG Engine ready for FastAPI")

class Email(BaseModel):
    subject: str
    body: str
    sender: str
    timestamp: str

SIGNATURE_PATTERNS = [
    r"(?i)(regards|thanks|sincerely|best|cheers)[,\s].*",
    r"(?i)--\s*\n.*",
    r"(?i)(sent from|get outlook).*",
]

DISCLAIMER_PATTERNS = [
    r"(?i)this email.*confidential.*",
    r"(?i)disclaimer:.*",
]

def clean_body(text: str) ->str:
    text = re.sub(r"<[^>]+>", " ", text)

    for s in SIGNATURE_PATTERNS:
        text=re.sub(s," ",text,flags=re.DOTALL)
    for d in DISCLAIMER_PATTERNS:
        text=re.sub(d," ",text,flags=re.DOTALL) 
    return text.strip()

def extract_features(text:str)->dict:
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["customer_name"] = ent.text
        elif ent.label_ == "ORG":
            entities["organization"] = ent.text

    vin = re.search(r'\b[A-HJ-NPR-Z0-9]{17}\b', text)
    if vin:
        entities["vin"] = vin.group()

    order_id = re.search(r'\b(ORD|ORDER|INV|REF)[#\-]?\d{4,10}\b', text, re.I)
    if order_id:
        entities["order_id"] = order_id.group()

    sku = re.search(r'\bSKU[- ]?[A-Z0-9]{4,12}\b', text)
    if sku:
        entities["product_sku"] = sku.group()

    return entities

def classify_intent(text):
    labels = [
        "warranty claim",
        "service issue",
        "billing issue",
        "shipping issue",
        "return or refund request",
        "product inquiry",
        "general enquiry"
    ]

    result = classifier(text, labels)

    return {
        "intent": result["labels"][0],
        "confidence": result["scores"][0]
    }


DEPARTMENT_LABELS = [
    "electronics",
    "appliances and kitchen",
    "clothing and fashion",
    "furniture",
    "groceries",
    "beauty and personal care",
    "sports and outdoors",
    "automotive",
    "toys and baby",
]

DEPT_KEYWORDS = {
    "electronics": ["laptop", "phone", "mobile", "tv", "television", "headphone", "earbud", "camera", "tablet", "charger"],
    "appliances and kitchen": ["blender", "mixer", "microwave", "fridge", "refrigerator", "oven", "cooktop", "dishwasher", "toaster", "pressure cooker", "utensil"],
    "clothing and fashion": ["shirt", "jeans", "dress", "t-shirt", "saree", "kurta", "trouser", "jacket", "shoe", "sneaker", "sandals"],
    "furniture": ["sofa", "chair", "table", "desk", "bed", "mattress", "wardrobe", "couch"],
    "groceries": ["rice", "flour", "atta", "dal", "oil", "snack", "beverage", "grocery", "milk", "bread"],
    "beauty and personal care": ["cream", "lotion", "shampoo", "conditioner", "lipstick", "makeup", "soap", "perfume", "razor"],
    "sports and outdoors": ["bat", "ball", "racket", "helmet", "cycle", "bicycle", "treadmill", "dumbbell"],
    "automotive": ["car", "bike", "motorcycle", "helmet", "tyre", "tire", "engine", "wiper", "horn"],
    "toys and baby": ["toy", "lego", "doll", "puzzle", "baby", "diaper", "stroller", "pram"],
}

def classify_department(text: str) -> dict:
    """Zero-shot department guess with keyword boost."""
    result = classifier(text, DEPARTMENT_LABELS)
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    text_lower = text.lower()
    for dept, kws in DEPT_KEYWORDS.items():
        if any(k in text_lower for k in kws):
            # light keyword boost if the zero-shot top_score is moderate
            if top_score < 0.7:
                return {"department": dept, "confidence": max(top_score, 0.65)}
            break

    return {"department": top_label, "confidence": top_score}


CRITICAL_KEYWORDS = [
    "accident",
    "damage",
    "broken",
    "malfunction",
    "faulty",
    "injury",
    "fire",
    "leak",
    "claim",
    "lawsuit",
    "legal",
    "escalate",
    "complaint",
    "chargeback",
    "fraud",
    "defect",
    "defective",
    "missing",
    "lost",
    "not received",
    "urgent",
    "asap",
    "immediately",
]

QUESTION_PHRASES = [
    "how",
    "when",
    "where",
    "what",
    "which",
    "can i",
    "do you",
    "is there",
    "are there",
    "does it",
    "policy",
]


def determine_routing(cleaned_text: str, intent: str, confidence: float) -> dict:
    text_lower = cleaned_text.lower()

    keyword_hit = any(k in text_lower for k in CRITICAL_KEYWORDS)
    question_hit = any(q in text_lower for q in QUESTION_PHRASES)

    if intent == "general enquiry" and confidence >= 0.4 and not keyword_hit:
        return {"route": "general", "reason": "general enquiry with sufficient confidence"}

    if intent in {"billing issue", "service issue", "warranty claim", "shipping issue", "product inquiry"} and confidence >= 0.4 and not keyword_hit:
        return {"route": "general", "reason": "known issue category with sufficient confidence"}

    if intent == "return or refund request" and confidence >= 0.4 and not keyword_hit:
        return {"route": "general", "reason": "return/refund with sufficient confidence and no risk terms"}

    if question_hit and not keyword_hit:
        return {"route": "general", "reason": "informational question without risk terms"}

    return {"route": "human", "reason": "potentially complex or critical content"}

def get_preprocessed_data(email: Email) -> dict:
    """Internal helper for email preprocessing logic."""
    cleaned = clean_body(email.body)
    
    try:
        lang = langdetect.detect(cleaned)
    except:
        lang = "unknown"
    
    entities = extract_features(cleaned)
    intent_result = classify_intent(cleaned)
    routing = determine_routing(cleaned, intent_result["intent"], intent_result["confidence"])
    dept_result = classify_department(cleaned) if routing["route"] == "human" else {"department": None, "confidence": None}
    
    return {
        "cleaned_body": cleaned,
        "language": lang,
        "entities": entities,
        "intent": intent_result["intent"],
        "confidence": intent_result["confidence"],
        "route": routing["route"],
        "route_reason": routing["reason"],
        "department": dept_result["department"],
        "department_confidence": dept_result["confidence"],
        "original_sender": email.sender,
        "subject": email.subject,
    }

@app.post("/preprocess")
def preprocess(email: Email):
    data = get_preprocessed_data(email)
    return JSONResponse(content=data)

@app.post("/generate-reply")
async def generate_reply(email: Email):
    # 1. Pipeline: Preprocess first
    prep = get_preprocessed_data(email)
    
    # 2. Logic: If route is human, don't auto-reply
    if prep["route"] == "human":
        return {
            "preprocess": prep,
            "reply": "Escalated to human support: " + prep["route_reason"],
            "status": "escalated"
        }
    
    # 3. RAG: Generate response
    name = prep["entities"].get("customer_name", "Valued Customer")
    rag_result = await rag_engine.get_response(
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

