"""Cal.com scheduling node.

Voice mode runs an explicit `BookingStage` machine: the caller's position in the booking
conversation comes from `state["booking_stage"]`, never from string-matching the assistant's
own previous message. Chat mode is a separate one-shot path because the browser has a real
booking form. Everything except the two Cal.com calls is a pure helper, so the stage handlers
are testable without a live client.
"""

import logging
import re
from datetime import datetime
from typing import Any, cast

from rag_persona.config import Settings
from rag_persona.schemas import BookingRequest, BookingStage, PersonaState
from rag_persona.services.calcom import CalComClient

logger = logging.getLogger(__name__)

MAX_OFFERED_SLOTS = 5
DEFAULT_ATTENDEE_NAME = "Recruiter"

REJECTION_MARKERS = ("no", "nope", "does not work", "doesn't work", "cannot do", "can't do",
                     "other time", "different time", "another time", "next")
_NAME_FILLER_RE = re.compile(r"^(yeah|yes|sure|okay|hi|hello|hey|uhm|uh|ah|great)\b[\s.,!?]*", re.I)
_NAME_PREFIXES = ("my name is", "this is", "i am", "sure, my name is", "sure, this is", "it is",
                  "its")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}")
_EMAIL_PREFIXES = ("my email is", "email is", "send to", "address is", "email to",
                   "email address is")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_HOUR_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
               8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def format_ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def _parse_slot(slot: str) -> datetime | None:
    try:
        return datetime.fromisoformat(slot.replace("Z", "+00:00"))
    except Exception:
        return None


def to_spoken_slot(slot_str: str) -> str:
    dt = _parse_slot(slot_str)
    if dt is None:
        return slot_str
    time_str = dt.strftime("%I:%M %p").lstrip("0")
    if time_str.endswith((":00 AM", ":00 PM")):
        time_str = time_str.replace(":00", "")
    return f"{dt.strftime('%A, %B')} {format_ordinal(dt.day)} at {time_str}"


def normalize_spoken_time(text: str) -> str:
    """Normalize time text (word numbers/ordinals to digits) for robust comparison."""
    mapping = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
               "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "eighth": "8th", "nine": "9", "ten": "10",
               "eleven": "11", "twelve": "12", "thirty": "30", "am": "am", "pm": "pm"}
    cleaned = re.sub(r"[^a-z0-9]", " ", text.lower())
    return " ".join(mapping.get(token, token) for token in cleaned.split())


def detect_slot_selection(raw_input: str, slots: list[str]) -> str | None:
    """Detect whether the caller named one of the offered slots outright."""
    if not slots:
        return None
    clean_input = raw_input.lower().strip()

    # 1. Ordinal references ("first", "second", ...).
    for idx, ord_word in enumerate(["first", "second", "third", "fourth", "fifth"]):
        if ord_word in clean_input and idx < len(slots):
            return slots[idx]

    # 2. Numeric choice references ("option two", "number one", ...).
    for idx, word in enumerate(["one", "two", "three", "four", "five"]):
        if idx < len(slots) and (f"option {word}" in clean_input
                                 or f"number {word}" in clean_input):
            return slots[idx]

    # 3. Weekday plus an exact time ("tuesday at 10:30", "monday at 10").
    for slot in slots:
        dt = _parse_slot(slot)
        if dt is None or dt.strftime("%A").lower() not in clean_input:
            continue
        hour = dt.strftime("%I").lstrip("0")
        if not (f" {hour} " in f" {clean_input} " or f" {dt.strftime('%-I')} " in f" {clean_input} "
                or f"{hour}:" in clean_input or _HOUR_WORDS.get(int(hour), "\0") in clean_input):
            continue
        half_past = "30" in clean_input or "thirty" in clean_input or "half past" in clean_input
        if (dt.minute == 0 and not half_past) or (dt.minute == 30 and half_past):
            return slot

    # 4. Weekday alone, when exactly one slot falls on that day.
    mentioned = [day for day in _WEEKDAYS if day in clean_input]
    if len(mentioned) == 1:
        days = [(s, _parse_slot(s)) for s in slots]
        matches = [s for s, dt in days if dt and dt.strftime("%A").lower() == mentioned[0]]
        if len(matches) == 1:
            return matches[0]
    return None


def is_rejection(text: str) -> bool:
    return any(marker in text.lower() for marker in REJECTION_MARKERS)


def extract_name(text: str) -> str:
    """Pull a name out of an utterance, stripping filler words and spoken lead-ins."""
    clean = _NAME_FILLER_RE.sub("", text.strip()).strip()
    lowered = clean.lower()
    for prefix in _NAME_PREFIXES:
        if lowered.startswith(prefix):
            clean = clean[len(prefix):].strip()
            break
    return clean.title()


def extract_email(text: str) -> str | None:
    """Extract an email address from a (frequently mangled) speech transcript."""
    clean = text.lower().strip().rstrip(".?!,")

    # Transcription workarounds, NOT dead code: speech-to-text renders spoken addresses as
    # "at the rate" / "third egg mail" / "dot". Deleting these breaks voice booking.
    clean = clean.replace("at the rate of", "@").replace("at the rate", "@")
    clean = clean.replace("third egg mail", "gmail")
    clean = clean.replace("[at]", "@").replace("[dot]", ".")
    clean = clean.replace(" at ", "@").replace(" dot ", ".")

    clean = re.sub(r"\s*@\s*", "@", clean)
    clean = re.sub(r"\s*\.\s*", ".", clean)

    # Retry with every space removed, after dropping any spoken lead-in.
    despaced = clean
    for prefix in _EMAIL_PREFIXES:
        if despaced.startswith(prefix):
            despaced = despaced[len(prefix):].strip()
    despaced = re.sub(r"\s+", "", despaced)

    match = _EMAIL_RE.search(clean)
    if match is None:
        fallback = _EMAIL_RE.search(despaced)
        return fallback.group(0).rstrip(".") if fallback else None

    # A one- or two-character local part means we matched the tail of a spelled-out
    # address ("j o h n at gmail dot com" -> "n@gmail.com"). Prefer the despaced read.
    if len(match.group(0).split("@")[0]) <= 2:
        spelled = _EMAIL_RE.search(despaced)
        if spelled and len(spelled.group(0).split("@")[0]) > 2:
            return spelled.group(0).rstrip(".")
    return match.group(0).rstrip(".")


def parse_slots(payload: object) -> list[str]:
    """Flatten a Cal.com availability payload into a date-ordered list of ISO timestamps."""
    slots_data: object = {}
    if isinstance(payload, dict):
        inner = payload.get("data", {})
        slots_data = inner.get("slots", {}) if isinstance(inner, dict) else payload.get("slots", {})

    slots: list[str] = []
    if isinstance(slots_data, dict):
        for _, time_slots in sorted(slots_data.items()):
            if not isinstance(time_slots, list):
                continue
            for slot in time_slots:
                if isinstance(slot, dict) and "time" in slot:
                    slots.append(str(slot["time"]))
                elif isinstance(slot, str):
                    slots.append(slot)
    return slots


def calendar_unavailable_message(mode: str, username: str) -> str:
    """The friendly cal.com-link fallback used whenever Cal.com cannot be reached."""
    if mode == "voice":
        return (f"I'm having a bit of trouble accessing the calendar right now. You can book "
                f"directly at cal.com/{username} — Tejasv has good availability and would love "
                f"to connect.")
    return (f"I'm having trouble accessing the calendar right now. "
            f"You can book directly at cal.com/{username}")


def _reply(state: PersonaState, answer: str, **updates: Any) -> PersonaState:
    """Build this turn's state update: the spoken answer plus any booking-state changes."""
    return cast(PersonaState, {**state, "answer": answer, **updates})


def _abandon(state: PersonaState, answer: str) -> PersonaState:
    """Leave the booking flow with a message and a clean slate, so a retry re-fetches."""
    return _reply(state, answer, booking_stage=BookingStage.idle, booking_slots=[],
                  booking_slot_index=0)


def handle_idle(state: PersonaState, slots: list[str]) -> PersonaState:
    """Entry: pin the fetched slots into state and offer the first one."""
    return _reply(state,
                  f"My next available slot is {to_spoken_slot(slots[0])}. Does that work for you?",
                  booking_stage=BookingStage.offering, booking_slots=slots,
                  booking_slot_index=0, available_slots=slots)


def _accept_slot(state: PersonaState, index: int) -> PersonaState:
    return _reply(state, "Great! Can I get your name first?",
                  booking_stage=BookingStage.awaiting_name, booking_slot_index=index)


def handle_offering(state: PersonaState, slots: list[str], username: str) -> PersonaState:
    """Interpret the reply against the slot that was actually offered."""
    raw_input = state.get("raw_input", "")
    index = state.get("booking_slot_index", 0)

    # An explicitly named slot ("the third one", "Tuesday at 10:30") outranks a rejection marker,
    # so "no, Tuesday at two works better" jumps instead of blindly skipping ahead.
    if (chosen := detect_slot_selection(raw_input, slots)) is not None:
        return _accept_slot(state, slots.index(chosen))
    if not is_rejection(raw_input):
        return _accept_slot(state, index)

    next_index = index + 1
    if next_index >= len(slots):
        return _abandon(state, f"Those are all the slots I have in the near future. You can check "
                               f"my full calendar at cal.com/{username} to find another time.")
    return _reply(state, f"No problem. How about {to_spoken_slot(slots[next_index])}?",
                  booking_stage=BookingStage.offering, booking_slots=slots,
                  booking_slot_index=next_index)


def handle_awaiting_name(state: PersonaState) -> PersonaState:
    name = extract_name(state.get("raw_input", "")) or DEFAULT_ATTENDEE_NAME
    return _reply(state,
                  f"Thanks, {name}. And what email address should I send the calendar "
                  f"invitation to?",
                  booking_stage=BookingStage.awaiting_email, booking_name=name)


async def handle_awaiting_email(state: PersonaState, calcom: CalComClient,
                                username: str) -> PersonaState:
    email = extract_email(state.get("raw_input", ""))
    if email is None:
        # Stay put and re-prompt: the stored name and slot index survive untouched.
        return _reply(state, "I'm sorry, I didn't quite catch the email address. Could you please "
                             "spell it out or state it again?",
                      booking_stage=BookingStage.awaiting_email)

    slots = state.get("booking_slots", [])
    index = state.get("booking_slot_index", 0)
    if not slots or index >= len(slots):
        return _abandon(state, f"I'm sorry, I lost track of which slot you selected. "
                               f"You can book directly at cal.com/{username}.")

    slot = slots[index]
    spoken_date = to_spoken_slot(slot)
    try:
        await calcom.create_booking(BookingRequest(
            preferred_time=slot,
            attendee_name=state.get("booking_name") or DEFAULT_ATTENDEE_NAME,
            attendee_email=email,
            notes=None,
        ))
    except Exception:
        logger.exception("Failed to create Cal.com booking")
        return _abandon(state, f"I ran into an issue finalizing the booking on the calendar. "
                               f"However, I have saved your preference for {spoken_date} at "
                               f"{email}. You can also visit cal.com/{username} to secure it.")

    return _reply(state,
                  f"Perfect! I've booked our meeting for {spoken_date} and sent the calendar "
                  f"invitation to {email}. You're all set! Is there anything else I can help "
                  f"you with?",
                  booking_stage=BookingStage.confirmed, booking_email=email)


def handle_confirmed(state: PersonaState) -> PersonaState:
    """The meeting is already booked — acknowledge, never rebook."""
    email = state.get("booking_email", "")
    return _reply(state,
                  f"We're all set — that meeting is already on the calendar and the invitation is "
                  f"on its way{f' to {email}' if email else ''}. Is there anything else I can "
                  f"help you with?",
                  booking_stage=BookingStage.confirmed)


async def _voice_turn(state: PersonaState, calcom: CalComClient, username: str) -> PersonaState:
    stage = state.get("booking_stage", BookingStage.idle)
    if stage == BookingStage.confirmed:
        return handle_confirmed(state)
    if stage == BookingStage.awaiting_name:
        return handle_awaiting_name(state)
    if stage == BookingStage.awaiting_email:
        return await handle_awaiting_email(state, calcom, username)

    # idle and offering both need the slot list. Fetch it ONCE per call and reuse it: re-fetching
    # each turn lets availability shift underneath "the second one" mid-conversation.
    slots = state.get("booking_slots") or []
    if not slots:
        slots = parse_slots(await calcom.get_available_slots())[:MAX_OFFERED_SLOTS]
    if not slots:
        return _abandon(state, f"I couldn't find any available slots in the next seven days. "
                               f"You can check my calendar directly at cal.com/{username}.")
    if stage == BookingStage.offering:
        return handle_offering(state, slots, username)
    return handle_idle(state, slots)


async def _chat_turn(state: PersonaState, calcom: CalComClient, username: str) -> PersonaState:
    """Chat mode skips the stage machine entirely — the frontend has a real booking form."""
    slots = parse_slots(await calcom.get_available_slots())[:MAX_OFFERED_SLOTS]
    formatted = [dt.strftime("%A, %B %d, %Y at %I:%M %p") if (dt := _parse_slot(s)) else s
                 for s in slots]
    slots_list = "\n".join(f"- {s}" for s in formatted)
    return _reply(state,
                  "Here are my next available slots. Please choose one and click to book:\n\n"
                  f"{slots_list}\n\n"
                  f"Or you can visit my booking page directly: https://cal.com/{username}",
                  available_slots=slots)


async def calcom_node(state: PersonaState, settings: Settings,
                      calcom: CalComClient | None) -> PersonaState:
    username = settings.calcom_username or "tejasv"
    mode = state.get("mode", "chat")
    if calcom is None or not calcom.configured:
        return _reply(state, calendar_unavailable_message(mode, username))
    try:
        if mode == "voice":
            return await _voice_turn(state, calcom, username)
        return await _chat_turn(state, calcom, username)
    except Exception:
        logger.exception("Cal.com scheduling turn failed")
        return _reply(state, calendar_unavailable_message(mode, username))
