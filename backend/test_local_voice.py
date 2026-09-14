import json
import os
import sys

import httpx

# Default query or user-specified argument
query = "Tell me about Tejasv's education."
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])

url = os.getenv("VOICE_URL", "http://127.0.0.1:8000/voice")

# /voice requires this header whenever the backend has VAPI_WEBHOOK_SECRET set.
secret = os.getenv("VAPI_WEBHOOK_SECRET", "")
headers = {"x-vapi-secret": secret} if secret else {}

payload = {
    "message": {
        "type": "response-required",
        "messages": [
            {"role": "user", "content": query}
        ]
    }
}

print(f"Sending mock Vapi request to {url} ...")
print(f"Query: \"{query}\"\n")

try:
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=20.0) as r:
        if r.status_code == 401:
            print("401: set VAPI_WEBHOOK_SECRET to the same value the backend uses.")
        elif r.status_code != 200:
            print(f"Error: Server returned status code {r.status_code}")
            print(r.read().decode())
        else:
            print("Streamed Sentences (OpenAI SSE Format):")
            print("-" * 50)
            for line in r.iter_lines():
                if line.startswith("data: "):
                    content_str = line[6:].strip()
                    if content_str == "[DONE]":
                        print("[assistant]: [DONE]")
                        continue
                    try:
                        data = json.loads(content_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(f"[assistant]: {content.strip()}")
                    except Exception as parse_err:
                        print(f"Failed to parse line: {line} - {parse_err}")
            print("-" * 50)
except httpx.ConnectError:
    print("Error: Could not connect to the local FastAPI backend.")
    print("Start it with: uvicorn rag_persona.main:app --reload")
except Exception as e:
    print(f"An error occurred: {e}")
