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

    # 3. Match weekday and exact time (e.g. "tuesday at 10:30" or "monday at 10")
    for slot in slots:
        try:
            dt = datetime.fromisoformat(slot.replace("Z", "+00:00"))
            weekday = dt.strftime("%A").lower()
            if weekday not in clean_input:
                continue
                
            hour = dt.strftime("%I").lstrip("0")  # "10", "2", etc.
            hour_alt = dt.strftime("%-I")
            minute = dt.minute  # integer: 0, 30, etc.
            
            # Check if hour is mentioned as digit or in word form
            hour_mentioned = (
                f" {hour} " in f" {clean_input} " 
                or f" {hour_alt} " in f" {clean_input} " 
                or f"{hour}:" in clean_input
            )
            
            hour_words = {
                1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve"
            }
            h_int = int(hour)
            if h_int in hour_words and hour_words[h_int] in clean_input:
                hour_mentioned = True
                
            if not hour_mentioned:
                continue
                
            # Check if minute matches
            if minute == 0:
                # If slot is on the hour, user shouldn't mention minutes like "30" or "thirty"
                if "30" in clean_input or "thirty" in clean_input or "half past" in clean_input:
                    continue
                return slot
            elif minute == 30:
                if "30" in clean_input or "thirty" in clean_input or "half past" in clean_input:
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


def extract_email(text: str) -> str | None:
    """
    Extracts email addresses from spoken transcription input.
    """
    import re
    # Clean text to handle common verbal formats of emails
    clean = text.lower().strip()
    clean = clean.replace("[at]", "@").replace("[dot]", ".").replace(" at ", "@").replace(" dot ", ".")
    
    # A standard email regex search on the spaced string first, to isolate it from prefixes/suffixes
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", clean)
    if match:
        return match.group(0)
    
    # If not matched (e.g. spaces inside the email itself like "john . doe @ gmail . com")
    # we remove all spaces and try again, but strip common prefix phrases first
    prefixes = ["my email is", "email is", "send to", "address is", "email to", "email address is"]
    temp = clean
    for prefix in prefixes:
        if temp.startswith(prefix):
            temp = temp[len(prefix):].strip()
            
    spaced_clean = re.sub(r'\s+', '', temp)
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", spaced_clean)
    if match:
        return match.group(0)
        
    return None



async def calcom_node(
    state: PersonaState,
    settings,
    calcom: CalComClient | None,
) -> PersonaState:
    username = settings.calcom_username or "tejasv"
    mode = state.get("mode", "chat")
    raw_input = state.get("raw_input", "")

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
        # Fetch slots from Cal.com
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

        # Handle voice email collection and booking
        if mode == "voice":
            history = state.get("conversation_history", [])
            expecting_email = False
            if history:
                last_assistant_msg = ""
                for msg in reversed(history):
                    if msg.get("role") == "assistant":
                        last_assistant_msg = msg.get("content", "")
                        break
                if last_assistant_msg and ("email address" in last_assistant_msg.lower() or "what email" in last_assistant_msg.lower() or "send the invitation to" in last_assistant_msg.lower() or "send the invite to" in last_assistant_msg.lower()):
                    expecting_email = True

            if expecting_email:
                email = extract_email(raw_input)
                if email:
                    # Retrieve the selected slot from conversation history
                    selected_slot = None
                    for msg in reversed(history):
                        if msg.get("role") == "assistant" and "I can book" in msg.get("content", ""):
                            idx = history.index(msg)
                            for j in range(idx - 1, -1, -1):
                                if history[j].get("role") == "user":
                                    selected_slot = detect_slot_selection(history[j].get("content", ""), slots)
                                    if selected_slot:
                                        break
                            if selected_slot:
                                break
                    
                    # Fallback to first available if not found
                    if not selected_slot and slots:
                        selected_slot = slots[0]

                    if selected_slot:
                        from rag_persona.schemas import BookingRequest
                        spoken_date = to_spoken_slot(selected_slot)
                        try:
                            req = BookingRequest(
                                preferred_time=selected_slot,
                                attendee_name="Phone Recruiter",
                                attendee_email=email,
                                notes="Booked via Portfolio Voice Agent."
                            )
                            await calcom.create_booking(req)
                            answer = f"Perfect! I've booked our meeting for {spoken_date} and sent the calendar invitation to {email}. You should receive it in a few moments. Is there anything else I can help you with?"
                            return {
                                **state,
                                "answer": answer,
                                "selected_slot": selected_slot,
                                "booking_confirmed": True
                            }
                        except Exception as e:
                            logger.exception("Failed to create Cal.com booking")
                            answer = f"I ran into an issue finalizing the booking on the calendar. However, I have saved your preference for {spoken_date} at {email}. You can also visit cal.com/{username} to secure it."
                            return {
                                **state,
                                "answer": answer,
                            }
                    else:
                        answer = "I'm sorry, I lost track of which slot you selected. Which time works best for you again?"
                        return {
                            **state,
                            "answer": answer,
                        }
                else:
                    answer = "I'm sorry, I didn't quite catch the email address. Could you please spell it out or state it again?"
                    return {
                        **state,
                        "answer": answer,
                    }

            # Check if user is selecting a slot
            selected = detect_slot_selection(raw_input, slots)
            if selected:
                spoken_date = to_spoken_slot(selected)
                answer = f"Great! I can book {spoken_date} for you. What email address should I send the calendar invitation to?"
                return {
                    **state,
                    "answer": answer,
                    "selected_slot": selected,
                }
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
