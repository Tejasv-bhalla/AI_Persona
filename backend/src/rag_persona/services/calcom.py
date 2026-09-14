from datetime import UTC, datetime, timedelta

import httpx

from rag_persona.config import Settings
from rag_persona.schemas import BookingRequest


class CalComClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    async def aclose(self) -> None:
        await self.client.aclose()

    @property
    def configured(self) -> bool:
        return bool(
            self.settings.calcom_api_key
            and self.settings.calcom_event_type_id
            and self.settings.calcom_username
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.calcom_api_key}",
            "cal-api-version": "2024-08-13",
            "Content-Type": "application/json",
        }

    async def get_available_slots(self) -> dict[str, object]:
        if not self.configured:
            raise RuntimeError("Cal.com credentials are not configured")

        now = datetime.now(UTC)
        start_time = now.isoformat()
        end_time = (now + timedelta(days=7)).isoformat()

        params: dict[str, str | int] = {
            "eventTypeId": int(self.settings.calcom_event_type_id),
            "startTime": start_time,
            "endTime": end_time,
            "timeZone": "Asia/Kolkata",
        }

        response = await self.client.get(
            "https://api.cal.com/v2/slots/available",
            params=params,
            headers=self._headers(),
        )
        response.raise_for_status()
        slots: dict[str, object] = response.json()
        return slots

    async def create_booking(self, request: BookingRequest) -> dict[str, object]:
        if not self.configured:
            raise RuntimeError("Cal.com credentials are not configured")

        payload = {
            "eventTypeId": int(self.settings.calcom_event_type_id),
            "start": request.preferred_time,
            "attendee": {
                "name": request.attendee_name,
                "email": request.attendee_email,
                "timeZone": "Asia/Kolkata",
            },
            "metadata": {},
        }

        response = await self.client.post(
            "https://api.cal.com/v2/bookings",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        booking: dict[str, object] = response.json()
        return booking
