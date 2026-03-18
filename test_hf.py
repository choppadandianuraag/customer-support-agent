from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=os.environ.get("HF_TOKEN")
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello!"}
]

print("Sending request...")
response = client.chat_completion(messages, max_tokens=500)
print(f"Full response: {response}")
print(f"Type: {type(response)}")
print(f"Content: {response.choices[0].message['content']}")
