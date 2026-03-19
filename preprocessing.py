"""
Email preprocessing module for the TechGear Customer Support Agent.

Handles email cleaning, entity extraction, intent classification,
department detection, and routing logic.
"""

import re
import langdetect
import asyncio

# --- Lazy globals (loaded on first request) ---
_classifier = None
_nlp = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        print("Loading zero-shot classifier...")
        _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _classifier


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        print("Loading spaCy model...")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ─── Constants ──────────────────────────────────────────────────────────────────

SIGNATURE_PATTERNS = [
    r"(?i)(regards|thanks|sincerely|best|cheers)[,\s].*",
    r"(?i)--\s*\n.*",
    r"(?i)(sent from|get outlook).*",
]

DISCLAIMER_PATTERNS = [
    r"(?i)this email.*confidential.*",
    r"(?i)disclaimer:.*",
]

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

CRITICAL_KEYWORDS = [
    "accident", "damage", "broken", "malfunction", "faulty",
    "injury", "fire", "leak", "claim", "lawsuit",
    "legal", "escalate", "complaint", "chargeback", "fraud",
    "defect", "defective", "missing", "lost", "not received",
    "urgent", "asap", "immediately",
]

QUESTION_PHRASES = [
    "how", "when", "where", "what", "which",
    "can i", "do you", "is there", "are there", "does it", "policy",
]


# ─── Functions ──────────────────────────────────────────────────────────────────

def clean_body(text: str) -> str:
    """Remove HTML tags, signatures, and disclaimers from email body."""
    text = re.sub(r"<[^>]+>", " ", text)

    for s in SIGNATURE_PATTERNS:
        text = re.sub(s, " ", text, flags=re.DOTALL)
    for d in DISCLAIMER_PATTERNS:
        text = re.sub(d, " ", text, flags=re.DOTALL)
    return text.strip()


def extract_features(text: str) -> dict:
    """Extract named entities and identifiers (order ID, SKU, VIN) from text."""
    doc = get_nlp()(text)
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


def classify_intent(text: str) -> dict:
    """Classify the customer's intent using zero-shot classification."""
    labels = [
        "warranty claim",
        "service issue",
        "billing issue",
        "shipping issue",
        "return or refund request",
        "product inquiry",
        "general enquiry"
    ]

    result = get_classifier()(text, labels)

    return {
        "intent": result["labels"][0],
        "confidence": result["scores"][0]
    }


def classify_department(text: str) -> dict:
    """Zero-shot department guess with keyword boost."""
    result = get_classifier()(text, DEPARTMENT_LABELS)
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


def determine_routing(cleaned_text: str, intent: str, confidence: float) -> dict:
    """Decide whether an email should be auto-replied or escalated to a human."""
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


def get_preprocessed_data(email) -> dict:
    """
    Full preprocessing pipeline for an incoming email.

    Accepts an Email pydantic model (with subject, body, sender, timestamp).
    Returns a dict with cleaned body, language, entities, intent, routing, and department.
    """
    cleaned = clean_body(email.body)

    try:
        lang = langdetect.detect(cleaned)
    except Exception:
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
