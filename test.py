from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="Qwen/Qwen3.5-9B",
    token=os.environ.get("HF_TOKEN")
)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Explain RAG in simple terms."}
]

response = client.chat_completion(messages)

print(response.choices[0].message["content"])
