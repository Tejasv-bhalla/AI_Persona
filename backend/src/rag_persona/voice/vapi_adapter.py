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
    Aggregates a stream of tokens.
    Uses Fast-Start optimization: streams the first 4 words immediately to Vapi
    so the assistant starts speaking instantly, then streams the rest sentence-by-sentence.
    """
    buffer = ""
    # Matches complete sentences, handling decimal points (e.g. 8.5) and abbreviations/domains (e.g. cal.com) without splitting
    sentence_end_regex = re.compile(r"^((?:[^.!?\n]|[.!?](?!\s))*?[.!?]+(?=\s))")

    is_first_chunk = True
    fast_start_word_count = 4

    async for token in token_stream:
        buffer += token

        # Fast-start: yield the first few words immediately
        if is_first_chunk:
            space_indices = [i for i, char in enumerate(buffer) if char == ' ']
            if len(space_indices) >= fast_start_word_count:
                split_idx = space_indices[fast_start_word_count - 1]
                first_part = buffer[:split_idx + 1]
                buffer = buffer[split_idx + 1:]

                chunk = {
                    "choices": [
                        {
                            "delta": {
                                "content": first_part
                            }
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                is_first_chunk = False

        # Sentence-level streaming for the rest of the conversation
        while not is_first_chunk:
            # Find the first complete sentence in the buffer
            match = sentence_end_regex.search(buffer)
            if not match:
                break

            sentence = match.group(1)
            buffer = buffer[match.end():]

            clean_sentence = sentence.strip()
            if clean_sentence:
                chunk = {
                    "choices": [
                        {
                            "delta": {
                                "content": clean_sentence + " "
                            }
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

    # Handle any remaining text left in buffer
    remaining = buffer.strip()
    if remaining:
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": remaining + " "
                    }
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final OpenAI stream done signal
    yield "data: [DONE]\n\n"


