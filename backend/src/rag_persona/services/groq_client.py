import json
from collections.abc import AsyncIterator
from typing import cast

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam

from rag_persona.config import Settings

# Malformed model output can be arbitrarily long; keep enough to debug, not enough to flood logs.
_MAX_ERROR_CONTENT_CHARS = 500


class GroqClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        self.settings = settings
        self.client = AsyncGroq(api_key=settings.groq_api_key)

    @staticmethod
    def _build_messages(
        system: str,
        user: str,
        history: list[dict[str, str]] | None,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
        if history:
            # History arrives off the wire as plain {"role", "content"} dicts (ChatRequest),
            # which is the shape the SDK's message TypedDicts describe.
            messages.extend(cast(list[ChatCompletionMessageParam], history))
        messages.append({"role": "user", "content": user})
        return messages

    def _fallback_model_for(self, model: str) -> str | None:
        """Configured fallback, or None when there is nothing useful to fall back to."""
        fallback = self.settings.groq_fallback_model
        # Retrying the same model would just reproduce the same failure.
        if not fallback or fallback == model:
            return None
        return fallback

    async def json_completion(
        self,
        model: str,
        system: str,
        user: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        messages = self._build_messages(system, user, history)

        try:
            response = await self.client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=self.settings.max_output_tokens_json,
                response_format={"type": "json_object"},
                messages=messages,
            )
        except Exception as primary_error:
            fallback_model = self._fallback_model_for(model)
            if fallback_model is None:
                raise
            try:
                response = await self.client.chat.completions.create(
                    model=fallback_model,
                    temperature=0,
                    # Double the cap: the fallback is a reasoning model, which spends
                    # output tokens thinking before it emits any JSON and returns empty
                    # at the primary cap. This only runs after a primary failure, so the
                    # extra reservation is a rare event rather than steady-state budget.
                    max_tokens=self.settings.max_output_tokens_json * 2,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
            except Exception as fallback_error:
                # The primary failure is the useful one; the fallback failure is context.
                raise primary_error from fallback_error

        content = response.choices[0].message.content or "{}"
        try:
            parsed: dict[str, object] = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            # Callers (guard, grader) handle this; just make it legible before it propagates.
            raise json.JSONDecodeError(
                f"{e.msg} (model={model}, content={content[:_MAX_ERROR_CONTENT_CHARS]!r})",
                e.doc,
                e.pos,
            ) from e

    async def _stream_tokens(
        self,
        model: str,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=model,
            temperature=0.2,
            stream=True,
            max_tokens=max_tokens,
            messages=messages,
        )
        async for chunk in stream:
            # Reasoning models also emit `delta.reasoning`; only `content` is the answer.
            token = chunk.choices[0].delta.content
            if token:
                yield token

    async def stream_completion(
        self,
        model: str,
        system: str,
        user: str,
        history: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(system, user, history)
        limit = max_tokens or self.settings.max_output_tokens_chat

        emitted = False
        try:
            async for token in self._stream_tokens(model, messages, limit):
                emitted = True
                yield token
            return
        except Exception as e:
            fallback_model = self._fallback_model_for(model)
            # Emitted tokens are already on the caller's screen (or spoken by TTS), so a
            # fallback would replay the answer from the start. A truncated answer is
            # recoverable; a duplicated one is not.
            if emitted or fallback_model is None:
                raise
            primary_error = e

        try:
            async for token in self._stream_tokens(fallback_model, messages, limit):
                yield token
        except Exception as fallback_error:
            raise primary_error from fallback_error
