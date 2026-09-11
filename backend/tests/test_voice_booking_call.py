"""The voice booking state machine, driven turn-by-turn through the compiled graph.

Each turn is a separate `astream` call with a fresh input state, so anything the
flow remembers has to come back out of the checkpointer.
"""

from conftest import OFFERED_SLOTS, FakeCalcom, GraphRun

from rag_persona.schemas import BookingStage


def test_booking_stage_advances_across_separate_invocations(
    booking_call: tuple[GraphRun, FakeCalcom],
) -> None:
    run, _ = booking_call
    assert [turn.stage for turn in run.turns] == [
        BookingStage.offering,  # opener: first slot offered
        BookingStage.offering,  # rejection: next slot offered
        BookingStage.awaiting_name,  # acceptance
        BookingStage.awaiting_email,  # name captured
        BookingStage.awaiting_email,  # unparseable email: stays put
        BookingStage.confirmed,  # email captured and booked
    ]


def test_every_booking_turn_routes_to_scheduling_despite_a_rag_verdict(
    booking_call: tuple[GraphRun, FakeCalcom],
) -> None:
    """The guard calls "yes that works" a factual question; booking state must override it."""
    run, _ = booking_call
    assert [turn.route for turn in run.turns] == ["scheduling"] * len(run.turns)


def test_calcom_availability_is_fetched_once_per_call(
    booking_call: tuple[GraphRun, FakeCalcom],
) -> None:
    """Refetching per turn would let availability shift underneath "the second one"."""
    _, calcom = booking_call
    assert calcom.fetches == 1


def test_the_booked_slot_is_the_one_that_was_offered_and_accepted(
    booking_call: tuple[GraphRun, FakeCalcom],
) -> None:
    """The caller rejected slot 0 and accepted slot 1; booking slots[0] would be wrong."""
    run, calcom = booking_call
    assert len(calcom.bookings) == 1
    assert calcom.bookings[0].preferred_time == OFFERED_SLOTS[1]
    assert [turn.slot_index for turn in run.turns] == [0, 1, 1, 1, 1, 1]


def test_attendee_details_survive_from_earlier_turns(
    booking_call: tuple[GraphRun, FakeCalcom],
) -> None:
    _, calcom = booking_call
    assert calcom.bookings[0].attendee_name == "John Carter"
    assert calcom.bookings[0].attendee_email == "john.carter@gmail.com"


def test_unparseable_email_reprompts_without_losing_name_or_slot(
    booking_call: tuple[GraphRun, FakeCalcom],
) -> None:
    run, _ = booking_call
    failed_turn = run.turns[4]
    assert failed_turn.stage == BookingStage.awaiting_email
    assert failed_turn.slot_index == 1
    assert "didn't quite catch the email" in failed_turn.answer


def test_confirmed_booking_is_persisted_in_the_checkpoint(
    booking_call: tuple[GraphRun, FakeCalcom],
) -> None:
    run, _ = booking_call
    assert run.persisted["booking_stage"] == BookingStage.confirmed
    assert run.persisted["booking_name"] == "John Carter"
    assert run.persisted["booking_slots"] == OFFERED_SLOTS
