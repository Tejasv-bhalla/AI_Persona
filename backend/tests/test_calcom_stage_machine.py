"""Stage transitions of the voice booking machine, one handler at a time.

The integration test in test_voice_booking_call.py covers the happy path through
the graph; these cover the branches that path never reaches.
"""

import asyncio
from typing import Any

from rag_persona.config import Settings
from rag_persona.nodes.calcom import (
    calcom_node,
    handle_awaiting_email,
    handle_awaiting_name,
    handle_confirmed,
    handle_idle,
    handle_offering,
)
from rag_persona.schemas import BookingStage, PersonaState

SLOTS = ["2026-09-08T09:00:00Z", "2026-09-08T10:30:00Z", "2026-09-09T14:00:00Z"]


class RecordingCalcom:
    """Records bookings; optionally fails the way a Cal.com outage would."""

    configured = True

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.bookings: list[Any] = []

    async def get_available_slots(self) -> dict[str, Any]:
        return {"data": {"slots": {"d": [{"time": slot} for slot in SLOTS]}}}

    async def create_booking(self, request: Any) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("cal.com is down")
        self.bookings.append(request)
        return {"id": "bk_1"}


def state(**overrides: Any) -> PersonaState:
    base: dict[str, Any] = {"raw_input": "", "mode": "voice", "answer": ""}
    base.update(overrides)
    return base  # type: ignore[return-value]


# --- offering --------------------------------------------------------------


def test_the_first_turn_offers_the_earliest_slot_and_pins_the_list() -> None:
    result = handle_idle(state(), SLOTS)
    assert result["booking_stage"] == BookingStage.offering
    assert result["booking_slot_index"] == 0
    assert result["booking_slots"] == SLOTS
    assert "Tuesday, September 8th at 9 AM" in result["answer"]


def test_a_rejection_offers_the_next_slot_without_leaving_the_offer_stage() -> None:
    result = handle_offering(state(raw_input="no that doesn't work", booking_slot_index=0),
                             SLOTS, "tejasv")
    assert result["booking_stage"] == BookingStage.offering
    assert result["booking_slot_index"] == 1
    assert "Tuesday, September 8th at 10:30 AM" in result["answer"]


def test_an_acceptance_moves_on_with_the_slot_that_was_offered() -> None:
    result = handle_offering(state(raw_input="yes that works", booking_slot_index=1),
                             SLOTS, "tejasv")
    assert result["booking_stage"] == BookingStage.awaiting_name
    assert result["booking_slot_index"] == 1


def test_naming_a_slot_outranks_a_rejection_marker_in_the_same_breath() -> None:
    """"no, Tuesday at 10:30 works better" must jump to that slot, not skip past it."""
    result = handle_offering(state(raw_input="no, tuesday at 10:30 works better",
                                   booking_slot_index=0), SLOTS, "tejasv")
    assert result["booking_stage"] == BookingStage.awaiting_name
    assert result["booking_slot_index"] == 1


def test_rejecting_the_last_slot_hands_over_the_booking_link_and_resets() -> None:
    result = handle_offering(state(raw_input="no", booking_slot_index=len(SLOTS) - 1),
                             SLOTS, "tejasv")
    assert result["booking_stage"] == BookingStage.idle
    assert result["booking_slots"] == []
    assert "cal.com/tejasv" in result["answer"]


# --- name ------------------------------------------------------------------


def test_a_captured_name_moves_the_call_to_the_email_question() -> None:
    result = handle_awaiting_name(state(raw_input="uhm, my name is John Carter"))
    assert result["booking_stage"] == BookingStage.awaiting_email
    assert result["booking_name"] == "John Carter"


def test_an_empty_name_falls_back_to_a_placeholder_rather_than_stalling() -> None:
    result = handle_awaiting_name(state(raw_input=""))
    assert result["booking_stage"] == BookingStage.awaiting_email
    assert result["booking_name"] == "Recruiter"


# --- email and booking -----------------------------------------------------


def test_an_unparseable_email_reprompts_without_losing_the_name_or_slot() -> None:
    calcom = RecordingCalcom()
    before = state(raw_input="uh I'm not sure", booking_name="John Carter",
                   booking_slots=SLOTS, booking_slot_index=1)
    result = asyncio.run(handle_awaiting_email(before, calcom, "tejasv"))  # type: ignore[arg-type]
    assert result["booking_stage"] == BookingStage.awaiting_email
    assert result["booking_name"] == "John Carter"
    assert result["booking_slot_index"] == 1
    assert result["booking_slots"] == SLOTS
    assert calcom.bookings == []


def test_a_captured_email_books_the_selected_slot_and_confirms() -> None:
    calcom = RecordingCalcom()
    before = state(raw_input="john dot carter at gmail dot com", booking_name="John Carter",
                   booking_slots=SLOTS, booking_slot_index=1)
    result = asyncio.run(handle_awaiting_email(before, calcom, "tejasv"))  # type: ignore[arg-type]
    assert result["booking_stage"] == BookingStage.confirmed
    assert result["booking_email"] == "john.carter@gmail.com"
    assert calcom.bookings[0].preferred_time == SLOTS[1]
    assert calcom.bookings[0].attendee_name == "John Carter"


def test_a_lost_slot_list_bails_out_instead_of_booking_a_guess() -> None:
    calcom = RecordingCalcom()
    before = state(raw_input="john at gmail dot com", booking_name="John",
                   booking_slots=[], booking_slot_index=0)
    result = asyncio.run(handle_awaiting_email(before, calcom, "tejasv"))  # type: ignore[arg-type]
    assert result["booking_stage"] == BookingStage.idle
    assert calcom.bookings == []
    assert "cal.com/tejasv" in result["answer"]


def test_a_calcom_outage_at_booking_time_keeps_the_caller_informed() -> None:
    calcom = RecordingCalcom(fail=True)
    before = state(raw_input="john at gmail dot com", booking_name="John",
                   booking_slots=SLOTS, booking_slot_index=0)
    result = asyncio.run(handle_awaiting_email(before, calcom, "tejasv"))  # type: ignore[arg-type]
    assert result["booking_stage"] == BookingStage.idle
    assert "john@gmail.com" in result["answer"]
    assert "cal.com/tejasv" in result["answer"]


def test_a_confirmed_booking_is_acknowledged_never_rebooked() -> None:
    result = handle_confirmed(state(booking_email="john@gmail.com"))
    assert result["booking_stage"] == BookingStage.confirmed
    assert "john@gmail.com" in result["answer"]


# --- node-level fallback ---------------------------------------------------


def test_an_unconfigured_calendar_hands_out_the_booking_link_without_raising() -> None:
    settings = Settings(_env_file=None, calcom_username="tejasv")
    result = asyncio.run(calcom_node(state(raw_input="book a call"), settings, None))
    assert "cal.com/tejasv" in result["answer"]
    assert "booking_stage" not in result
