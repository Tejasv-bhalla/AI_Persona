import logging
import httpx
from datetime import datetime
from rag_persona.config import Settings

logger = logging.getLogger(__name__)


def format_ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def to_spoken_date(slot_str: str) -> str:
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


async def send_booking_sms(to: str, slot_time: str, settings: Settings) -> bool:
    """
    Send an SMS using Twilio to the caller, containing a link to finalize the booking.
    """
    if not to:
        logger.warning("No recipient phone number provided for booking SMS.")
        return False

    sid = settings.twilio_sid
    token = settings.twilio_token
    from_number = settings.twilio_from
    username = settings.calcom_username or "tejasv"

    if not sid or not token or not from_number:
        logger.warning("Twilio credentials are not fully configured. SMS skipped.")
        return False

    # Format the slot time for the SMS text
    spoken_time = to_spoken_date(slot_time)
    
    # URL encoded prefill parameter for Cal.com (e.g. date=...)
    booking_url = f"https://cal.com/{username}?date={slot_time}"
    body = (
        f"Hi! Thanks for calling. I've reserved your slot for {spoken_time}. "
        f"Please click here to enter your email and confirm the booking: {booking_url}"
    )

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    auth = (sid, token)
    data = {
        "To": to,
        "From": from_number,
        "Body": body,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, auth=auth, data=data)
            if response.is_success:
                logger.info(f"Booking SMS successfully sent to {to} for slot {slot_time}")
                return True
            else:
                logger.error(f"Failed to send Twilio SMS. Status: {response.status_code}, Body: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Error occurred while sending Twilio SMS: {e}")
        return False
