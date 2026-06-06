import httpx
import json
import sys

# Default query or user-specified argument
query = "Tell me about Tejasv's education."
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])

url = "http://127.0.0.1:8000/voice"

payload = {
    "message": {
        "type": "response-required",
        "messages": [
            {"role": "user", "content": query}
        ]
    }
}

print(f"Sending mock Vapi request to local backend on port 8000...")
print(f"Query: \"{query}\"\n")

try:
    with httpx.stream("POST", url, json=payload, timeout=20.0) as r:
        if r.status_code != 200:
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
    print("Please make sure your server is running on http://127.0.0.1:8000 (run: uvicorn rag_persona.main:app --reload)")
except Exception as e:
    print(f"An error occurred: {e}")
