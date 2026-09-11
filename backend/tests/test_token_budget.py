"""Guards the Groq free-tier output-token budget.

Groq's free tier allows 1000 output tokens per minute and admits a request against
its `max_tokens` rather than its actual output. A single chat turn makes three
calls, so raising any one cap can push a turn over the ceiling and make every
request fail with a 429 that looks nothing like a quota problem.
"""

from pathlib import Path

from rag_persona.config import Settings

FREE_TIER_OUTPUT_TOKENS_PER_MINUTE = 1000


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_a_chat_turn_fits_in_the_free_tier_minute_budget() -> None:
    s = _settings()
    # guard (json) -> generation (stream) -> grader (json)
    turn = s.max_output_tokens_json + s.max_output_tokens_chat + s.max_output_tokens_json
    assert turn <= FREE_TIER_OUTPUT_TOKENS_PER_MINUTE, (
        f"a chat turn reserves {turn} output tokens, over the {FREE_TIER_OUTPUT_TOKENS_PER_MINUTE} "
        "free-tier limit; every request will 429"
    )


def test_a_voice_turn_fits_in_the_free_tier_minute_budget() -> None:
    s = _settings()
    # Voice skips the grader.
    turn = s.max_output_tokens_json + s.max_output_tokens_voice
    assert turn <= FREE_TIER_OUTPUT_TOKENS_PER_MINUTE


def test_every_generation_path_sets_a_cap() -> None:
    """No cap means Groq reserves the model default, which alone can exceed the limit."""
    s = _settings()
    for cap in (s.max_output_tokens_json, s.max_output_tokens_chat, s.max_output_tokens_voice):
        assert cap > 0


def test_voice_cap_covers_the_spoken_word_limit() -> None:
    """The voice prompt asks for at most 80 words; leave room for ~1.5 tokens per word."""
    s = _settings()
    assert s.max_output_tokens_voice >= int(s.voice_max_response_words * 1.5)


def test_both_json_paths_declare_a_cap() -> None:
    """A create() call with no max_tokens is admitted against the model default.

    The fallback retry omitted its cap once already, which made a guard failure fall
    through to a call that 429s for an unrelated reason.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "rag_persona" / "services" / "groq_client.py"
    ).read_text()
    creates = source.count("chat.completions.create(")
    caps = source.count("max_tokens=")
    assert caps == creates, f"{creates} create() calls but only {caps} set max_tokens"
