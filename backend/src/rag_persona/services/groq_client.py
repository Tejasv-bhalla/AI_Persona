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

        try:
            response = await self.client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as e:
            if "70b" in model.lower():
                fallback_model = self.settings.groq_guard_model or "llama-3.1-8b-instant"
                try:
                    response = await self.client.chat.completions.create(
                        model=fallback_model,
                        temperature=0,
                        response_format={"type": "json_object"},
                        messages=messages,
                    )
                except Exception:
                    raise e
            else:
                raise e

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

        use_fallback = False
        fallback_err = None
        try:
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
        except Exception as e:
            if "70b" in model.lower():
                use_fallback = True
                fallback_err = e
            else:
                raise e

        if use_fallback:
            fallback_model = self.settings.groq_guard_model or "llama-3.1-8b-instant"
            try:
                stream = await self.client.chat.completions.create(
                    model=fallback_model,
                    temperature=0.2,
                    stream=True,
                    messages=messages,
                )
                async for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield token
            except Exception:
                raise fallback_err

