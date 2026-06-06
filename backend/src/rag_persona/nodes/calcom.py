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


async def calcom_node(
    state: PersonaState,
    settings,
    calcom: CalComClient | None,
) -> PersonaState:
    username = settings.calcom_username or "tejasv"
    mode = state.get("mode", "chat")

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

