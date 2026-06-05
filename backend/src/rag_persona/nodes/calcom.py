import logging
from datetime import datetime

from rag_persona.schemas import PersonaState
from rag_persona.services.calcom import CalComClient

logger = logging.getLogger(__name__)


async def calcom_node(
    state: PersonaState,
    settings,
    calcom: CalComClient | None,
) -> PersonaState:
    username = settings.calcom_username or "tejasv"
    if calcom is None or not calcom.configured:
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
            return {
                **state,
                "answer": f"I couldn't find any available slots in the next 7 days. You can check my calendar directly at cal.com/{username}.",
            }

        formatted_slots = []
        for slot in slots:
            try:
                # 2026-06-05T10:00:00Z -> "Friday, June 05, 2026 at 10:00 AM"
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
            "available_slots": slots,  # We can inject this into state
        }

    except Exception:
        logger.exception("Failed to fetch slots from Cal.com")
        return {
            **state,
            "answer": f"I'm having trouble accessing the calendar right now. You can book directly at cal.com/{username}",
        }
