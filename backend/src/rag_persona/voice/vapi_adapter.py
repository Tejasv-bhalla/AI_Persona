import json
import re
from collections.abc import AsyncIterator
from typing import Any


def parse_vapi_request(payload: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """
    Parses Vapi's custom LLM request payload.
    Extracts the latest user message and conversation history.
    """
    if "messages" in payload:
        messages = payload["messages"]
    else:
        message_obj = payload.get("message", {})
        messages = message_obj.get("messages", [])

    current_input = ""
    history: list[dict[str, str]] = []

    # Find the last message where role is user
    user_messages = [m for m in messages if m.get("role") == "user"]
    if user_messages:
        current_input = user_messages[-1].get("content", "")

    # Build conversation history of all prior turns (excluding system prompt)
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})

    # Avoid repeating the current user query in the history
    if history and history[-1]["role"] == "user" and history[-1]["content"] == current_input:
        history.pop()

    return current_input, history


async def format_vapi_response_stream(token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    Aggregates a stream of tokens into complete sentences.
    Streams each sentence as an NDJSON line. Marks the final sentence with "stop": True.
    """
    buffer = ""
    # Matches complete sentences, handling decimal points (e.g. 8.5) and abbreviations/domains (e.g. cal.com) without splitting
    sentence_end_regex = re.compile(r"^((?:[^.!?\n]|[.!?](?!\s))*?[.!?]+(?=\s))")
    last_sentence = None

    async for token in token_stream:
        buffer += token
        while True:
            # Find the first complete sentence in the buffer
            match = sentence_end_regex.search(buffer)
            if not match:
                break

            sentence = match.group(1)
            buffer = buffer[match.end():]

            clean_sentence = sentence.strip()
            if clean_sentence:
                if last_sentence is not None:
                    # Yield deferred previous sentence
                    chunk = {"role": "assistant", "content": last_sentence}
                    yield json.dumps(chunk) + "\n"
                last_sentence = clean_sentence

    # Handle any remaining text left in buffer
    remaining = buffer.strip()
    if remaining:
        if last_sentence is not None:
            chunk = {"role": "assistant", "content": last_sentence}
            yield json.dumps(chunk) + "\n"
        last_sentence = remaining

    # Yield the final sentence with stop: True
    if last_sentence is not None:
        chunk = {"role": "assistant", "content": last_sentence, "stop": True}
        yield json.dumps(chunk) + "\n"
    else:
        # Graceful fallback if no text was generated
        chunk = {"role": "assistant", "content": "", "stop": True}
        yield json.dumps(chunk) + "\n"
