import logging
from datetime import datetime

from rag_persona.schemas import PersonaState
from rag_persona.services.calcom import CalComClient

logger = logging.getLogger(__name__)


def format_ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def to_spoken_slot(slot_str: str) -> str:
    try:
        dt = datetime.fromisoformat(slot_str.replace("Z", "+00:00"))
        day_str = format_ordinal(dt.day)
        time_str = dt.strftime("%I:%M %p").lstrip("0")
        if time_str.endswith(":00 AM"):
            time_str = time_str.replace(":00", "")
        elif time_str.endswith(":00 PM"):
            time_str = time_str.replace(":00", "")
        return f"{dt.strftime('%A, %B')} {day_str} at {time_str}"
    except Exception:
        return slot_str


def detect_slot_selection(raw_input: str, slots: list[str]) -> str | None:
    """
    Detect if the user selected one of the offered availability slots.
    """
    if not slots:
        return None

    clean_input = raw_input.lower().strip()

    # 1. Match ordinal references ("first", "second", "third", etc.)
    ordinals = ["first", "second", "third", "fourth", "fifth"]
    for idx, ord_word in enumerate(ordinals):
        if ord_word in clean_input and idx < len(slots):
            return slots[idx]

    # 2. Match numeric choice references ("option two", "number one", etc.)
    options = ["one", "two", "three", "four", "five"]
    for idx, opt_word in enumerate(options):
        if (f"option {opt_word}" in clean_input or f"number {opt_word}" in clean_input) and idx < len(slots):
            return slots[idx]

    # 3. Match weekday and hour (e.g. "tuesday at 10" or "monday at 2")
    for slot in slots:
        try:
            dt = datetime.fromisoformat(slot.replace("Z", "+00:00"))
            weekday = dt.strftime("%A").lower()
            hour = dt.strftime("%I").lstrip("0")
            hour_alt = dt.strftime("%-I")
            
            # If user mentions the day AND the hour digit
            if weekday in clean_input and (hour in clean_input or hour_alt in clean_input):
                return slot
        except Exception:
            continue

    # 4. Fallback match: if user mentions only the weekday and there is exactly one slot on that day
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    mentioned_days = [day for day in weekdays if day in clean_input]
    if len(mentioned_days) == 1:
        matches = []
        for slot in slots:
            try:
                dt = datetime.fromisoformat(slot.replace("Z", "+00:00"))
                if dt.strftime("%A").lower() == mentioned_days[0]:
                    matches.append(slot)
            except Exception:
                continue
        if len(matches) == 1:
            return matches[0]

    return None


async def calcom_node(
    state: PersonaState,
    settings,
    calcom: CalComClient | None,
) -> PersonaState:
    username = settings.calcom_username or "tejasv"
    mode = state.get("mode", "chat")
    raw_input = state.get("raw_input", "")

    # Handle clean slot selection logic for voice calls
    if mode == "voice" and state.get("available_slots"):
        selected = detect_slot_selection(raw_input, state["available_slots"])
        if selected:
            spoken_date = to_spoken_slot(selected)
            
            # Send Twilio SMS in the background
            import asyncio
            from rag_persona.services.sms import send_booking_sms
            asyncio.create_task(
                send_booking_sms(
                    to=state.get("customer_number", ""),
                    slot_time=selected,
                    settings=settings
                )
            )

            # Check if Twilio is configured to adapt confirmation text
            twilio_configured = bool(settings.twilio_sid and settings.twilio_token and settings.twilio_from)
            if twilio_configured:
                answer = (
                    f"Great! I've reserved {spoken_date} for you. "
                    "I just sent a text message to your phone with the booking link. "
                    "Simply tap it to enter your email and finalize. I look forward to speaking with you!"
                )
            else:
                answer = (
                    f"Great! I've reserved {spoken_date} for you. "
                    f"To finalize the booking, please go to cal.com/{username} to enter your email. "
                    "I look forward to speaking with you!"
                )

            return {
                **state,
                "answer": answer,
                "selected_slot": selected,
            }

    if calcom is None or not calcom.configured:
        if mode == "voice":
            return {
                **state,
                "answer": f"I'm having a bit of trouble accessing the calendar right now. You can book directly at cal.com/{username} — Tejasv has good availability and would love to connect.",
            }
        return {
            **state,
            "answer": f"I'm having trouble accessing the calendar right now. You can book directly at cal.com/{username}",
        }

    try:
        data = await calcom.get_available_slots()
        slots_data = {}
        if isinstance(data, dict):
            inner_data = data.get("data", {})
            if isinstance(inner_data, dict):
                slots_data = inner_data.get("slots", {})
            else:
                slots_data = data.get("slots", {})

        slots = []
        if isinstance(slots_data, dict):
            for _, time_slots in sorted(slots_data.items()):
                if isinstance(time_slots, list):
                    for slot in time_slots:
                        if isinstance(slot, dict) and "time" in slot:
                            slots.append(slot["time"])
                        elif isinstance(slot, str):
                            slots.append(slot)

        slots = slots[:5]
        if not slots:
            if mode == "voice":
                return {
                    **state,
                    "answer": f"I couldn't find any available slots in the next seven days. You can check my calendar directly at cal.com/{username}.",
                }
            return {
                **state,
                "answer": f"I couldn't find any available slots in the next 7 days. You can check my calendar directly at cal.com/{username}.",
            }

        if mode == "voice":
            spoken_slots = [to_spoken_slot(s) for s in slots[:3]]
            if len(spoken_slots) == 1:
                slots_phrase = spoken_slots[0]
            elif len(spoken_slots) == 2:
                slots_phrase = f"{spoken_slots[0]} or {spoken_slots[1]}"
            else:
                slots_phrase = f"{spoken_slots[0]}, {spoken_slots[1]}, or {spoken_slots[2]}"

            answer = (
                f"I have availability on {slots_phrase}. "
                "Which of those works best for you?"
            )
            return {
                **state,
                "answer": answer,
                "available_slots": slots,
            }

        formatted_slots = []
        for slot in slots:
            try:
                dt = datetime.fromisoformat(slot.replace("Z", "+00:00"))
                formatted_slots.append(dt.strftime("%A, %B %d, %Y at %I:%M %p"))
            except Exception:
                formatted_slots.append(slot)

        slots_list_str = "\n".join(f"- {s}" for s in formatted_slots)

        answer = (
            "Here are my next available slots. Please choose one and click to book:\n\n"
            f"{slots_list_str}\n\n"
            f"Or you can visit my booking page directly: https://cal.com/{username}"
        )

        return {
            **state,
            "answer": answer,
            "available_slots": slots,
        }

    except Exception:
        logger.exception("Failed to fetch slots from Cal.com")
        if mode == "voice":
            return {
                **state,
                "answer": f"I'm having a bit of trouble accessing the calendar right now. You can book directly at cal.com/{username} — Tejasv has good availability and would love to connect.",
            }
        return {
            **state,
            "answer": f"I'm having trouble accessing the calendar right now. You can book directly at cal.com/{username}",
        }
