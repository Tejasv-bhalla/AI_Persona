"""Which node a turn goes to, given the guard verdict and the booking state."""

from typing import Any

from rag_persona.nodes.router import route_from_state
from rag_persona.schemas import BookingStage, GuardResult, Intent, PersonaState, SafetyVerdict


def guard(intent: Intent, safety: SafetyVerdict = SafetyVerdict.safe) -> GuardResult:
    return GuardResult(safety=safety, intent=intent, keywords="k")


def state(**overrides: Any) -> PersonaState:
    return overrides  # type: ignore[return-value]


def test_an_unsafe_verdict_is_refused_before_anything_else_is_considered() -> None:
    unsafe = state(
        guard=guard(Intent.scheduling, SafetyVerdict.malicious),
        booking_stage=BookingStage.awaiting_email,
    )
    assert route_from_state(unsafe) == "refusal"
    assert route_from_state(state(guard=guard(Intent.rag, SafetyVerdict.suspicious))) == "refusal"


def test_a_turn_mid_booking_stays_in_scheduling_whatever_the_guard_decided() -> None:
    """"john at gmail dot com" reads as a factual question; it is an answer to our question."""
    for stage in (BookingStage.offering, BookingStage.awaiting_name, BookingStage.awaiting_email):
        assert route_from_state(state(guard=guard(Intent.rag), booking_stage=stage)) == (
            "scheduling"
        ), stage


def test_an_idle_or_finished_booking_does_not_capture_the_next_question() -> None:
    assert route_from_state(state(guard=guard(Intent.rag), booking_stage=BookingStage.idle)) == (
        "rag"
    )
    assert route_from_state(
        state(guard=guard(Intent.small_talk), booking_stage=BookingStage.confirmed)
    ) == "small_talk"


def test_intent_decides_the_route_when_no_booking_is_in_flight() -> None:
    assert route_from_state(state(guard=guard(Intent.rag))) == "rag"
    assert route_from_state(state(guard=guard(Intent.scheduling))) == "scheduling"
    assert route_from_state(state(guard=guard(Intent.small_talk))) == "small_talk"
    assert route_from_state(state(guard=guard(Intent.end_call))) == "end_call"
