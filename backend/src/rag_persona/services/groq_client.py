import json
from collections.abc import AsyncIterator

from groq import AsyncGroq

from rag_persona.config import Settings


class GroqClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        self.settings = settings
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    async def json_completion(
        self,
        model: str,
        system: str,
        user: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        response = await self.client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def stream_completion(
        self,
        model: str,
        system: str,
        user: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        stream = await self.client.chat.completions.create(
            model=model,
            temperature=0.2,
            stream=True,
            messages=messages,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token

