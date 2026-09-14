"""The Vapi custom-LLM bridge: request parsing and the SSE response stream.

The sentence splitter is load-bearing for the voice product — a bad split makes
the TTS engine pause mid-number or mid-domain — so it gets the most attention here.
"""

import asyncio
import json
from collections.abc import AsyncIterator

from rag_persona.voice.vapi_adapter import format_vapi_response_stream, parse_vapi_request


async def _emit(tokens: list[str]) -> AsyncIterator[str]:
    for token in tokens:
        yield token


def stream(tokens: list[str]) -> list[str]:
    """Run the formatter to completion and return the raw SSE lines."""

    async def collect() -> list[str]:
        return [line async for line in format_vapi_response_stream(_emit(tokens))]

    return asyncio.run(collect())


def contents(lines: list[str]) -> list[str]:
    """The spoken text of each SSE frame, excluding the terminator."""
    out: list[str] = []
    for line in lines:
        body = line.removeprefix("data: ").strip()
        if body == "[DONE]":
            continue
        out.append(json.loads(body)["choices"][0]["delta"]["content"])
    return out


WORDY = ["I ", "am ", "doing ", "well ", "today. ", "The ", "score ", "was ", "8.5 ", "out ",
         "of ", "10. ", "You ", "can ", "book ", "at ", "cal.com/tejasv ", "any ", "time."]


def test_the_first_frame_flushes_after_four_words_so_speech_starts_immediately() -> None:
    first = contents(stream(WORDY))[0]
    assert first == "I am doing well "


def test_after_the_fast_start_output_is_emitted_one_sentence_at_a_time() -> None:
    spoken = contents(stream(WORDY))
    assert spoken[1:] == [
        "today. ",
        "The score was 8.5 out of 10. ",
        "You can book at cal.com/tejasv any time. ",
    ]


def test_a_decimal_number_does_not_end_a_sentence() -> None:
    assert not any(frame.strip() == "8.5" for frame in contents(stream(WORDY)))
    assert any("8.5 out of 10." in frame for frame in contents(stream(WORDY)))


def test_a_domain_name_does_not_end_a_sentence() -> None:
    spoken = contents(stream(["Book ", "at ", "cal.com/tejasv ", "or ", "email ", "me ", "now."]))
    assert all("cal.com" not in frame or "cal.com/tejasv or" in frame for frame in spoken)


def test_the_stream_always_ends_with_the_openai_done_sentinel() -> None:
    assert stream(WORDY)[-1] == "data: [DONE]\n\n"
    assert stream(["Hi."])[-1] == "data: [DONE]\n\n"


def test_a_reply_shorter_than_the_fast_start_window_is_still_flushed() -> None:
    """Fewer than four words means the fast-start split never fires; the tail flush must."""
    assert contents(stream(["Sure ", "thing."])) == ["Sure thing. "]


def test_no_spoken_text_is_lost_between_input_tokens_and_emitted_frames() -> None:
    assert "".join(contents(stream(WORDY))).split() == "".join(WORDY).split()


def test_text_with_no_terminal_punctuation_is_still_spoken() -> None:
    tokens = ["Just ", "a ", "few ", "words ", "with ", "no ", "period"]
    assert "".join(contents(stream(tokens))).split() == "".join(tokens).split()


def test_the_newest_user_message_becomes_the_current_turn() -> None:
    payload = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "tell me about the stock market project"},
    ]}
    current, history = parse_vapi_request(payload)
    assert current == "tell me about the stock market project"


def test_history_excludes_the_current_turn_and_the_system_prompt() -> None:
    payload = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "tell me more"},
    ]}
    _, history = parse_vapi_request(payload)
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_the_nested_message_payload_shape_is_handled_identically() -> None:
    """Vapi sends `message.messages` on some webhook versions and `messages` on others."""
    payload = {"message": {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "tell me more"},
    ]}}
    assert parse_vapi_request(payload) == ("tell me more", [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])


def test_the_opening_turn_of_a_call_has_no_history() -> None:
    payload = {"message": {"messages": [{"role": "user", "content": "hello there"}]}}
    assert parse_vapi_request(payload) == ("hello there", [])


def test_a_payload_with_no_messages_yields_an_empty_turn_rather_than_raising() -> None:
    assert parse_vapi_request({}) == ("", [])
