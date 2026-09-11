"""The corrective-RAG retry loop, driven through the real compiled graph."""

from conftest import GradingGroq, GraphRun


def test_ungrounded_answer_is_regenerated_exactly_once(
    ungrounded_then_grounded: tuple[GraphRun, GradingGroq],
) -> None:
    _, groq = ungrounded_then_grounded
    assert groq.generation_calls == 2


def test_retry_widens_the_retrieved_context(
    ungrounded_then_grounded: tuple[GraphRun, GradingGroq],
) -> None:
    """A narrow context is the likeliest reason an answer could not be grounded."""
    _, groq = ungrounded_then_grounded
    first, second = groq.chunks_in_prompt
    assert second > first


def test_retried_answer_is_streamed_as_correction_not_token(
    ungrounded_then_grounded: tuple[GraphRun, GradingGroq],
) -> None:
    """The client has already rendered the first answer; it must replace, not append."""
    run, _ = ungrounded_then_grounded
    types = run.event_types()
    assert "correction" in types
    # The first pass streams `token`; everything after the retry starts is `correction`.
    assert types.index("correction") > types.index("token")
    assert types[-1] == "correction"


def test_second_answer_wins_and_is_reported_as_grounded(
    ungrounded_then_grounded: tuple[GraphRun, GradingGroq],
) -> None:
    run, _ = ungrounded_then_grounded
    assert run.final["grounded"] is True
    assert run.final["retry_count"] == 1
    assert run.final["answer"] == "He studied at IIT Roorkee."


def test_grounded_answer_is_not_regenerated(
    grounded_first_time: tuple[GraphRun, GradingGroq],
) -> None:
    run, groq = grounded_first_time
    assert groq.generation_calls == 1
    assert run.final["retry_count"] == 0
    assert "correction" not in run.event_types()
