import os
import uuid
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timezone, timedelta
# env var pointing to your service account JSON
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "scooter-474308-28615053db91.json")
DELEGATED_USER = os.getenv("DELEGATED_USER", "hello@thescooter.ai")

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events"
]

def get_calendar_service():
    if not SERVICE_ACCOUNT_FILE or not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise RuntimeError("Service account JSON not found. Set GOOGLE_SERVICE_ACCOUNT_FILE env var.")

    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    # IMPORTANT: impersonate a real workspace user — this requires DWD configured in Admin Console
    delegated_creds = creds.with_subject(DELEGATED_USER)

    service = build("calendar", "v3", credentials=delegated_creds, cache_discovery=False)
    return service

def create_meeting(student_email, mentor_email, start_time, end_time,
                   calendar_id: str = "primary", summary: str = "Interview",
                   description: Optional[str] = None, send_updates: str = "all"):
    """
    Create a Google Calendar event with a Google Meet link and invite student & mentor.
    - start_time and end_time should be datetime objects (with tzinfo ideally) or ISO strings.
    - DOES NOT check availability (will try to create event unconditionally).
    - Returns dict with created event and extracted meet link, or raises an exception.
    """
    # normalize times to ISO strings if they are datetime objects
    try:
        start_iso = start_time.isoformat() if hasattr(start_time, "isoformat") else str(start_time)
        end_iso = end_time.isoformat() if hasattr(end_time, "isoformat") else str(end_time)
    except Exception:
        raise ValueError("start_time and end_time must be datetime objects or ISO strings")

    service = get_calendar_service()

    event_body = {
        'summary': summary,
        'description': description or "",
        'start': {
            'dateTime': start_iso,
            'timeZone': 'Asia/Kolkata'
        },
        'end': {
            'dateTime': end_iso,
            'timeZone': 'Asia/Kolkata'
        },
        "organizer": {
    "email": "hello@thescooter.ai"
},
        'attendees': [{'email': student_email}, {'email': mentor_email}],
        'conferenceData': {
            'createRequest': {
                # unique requestId for conference creation
                'requestId': str(uuid.uuid4())
            }
        }
    }

    try:
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event_body,
            conferenceDataVersion=1,
            sendUpdates=send_updates
        ).execute()

        # extract Meet link
        meet_link = created_event.get('hangoutLink')
        if not meet_link:
            cd = created_event.get('conferenceData', {})
            for ep in cd.get('entryPoints', []) if cd else []:
                if ep.get('entryPointType') in ("video", "more"):
                    meet_link = ep.get('uri')
                    break

        return {
            "status": "created",
            "eventId": created_event.get("id"),
            "htmlLink": created_event.get("htmlLink"),
            "meetLink": meet_link,
            "raw": created_event
        }

    except HttpError as e:
        # bubble up Google API error with details
        raise RuntimeError(f"Google API error: {e.resp.status} - {e._get_reason() if hasattr(e, '_get_reason') else e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create event: {e}")

def create_event_without_meet(attendee_email: str,
                              start_time,
                              end_time,
                              calendar_id: str = "primary",
                              summary: str = "Interview",
                              description: Optional[str] = None,
                              send_updates: str = "all"):

    try:
        start_iso = start_time.isoformat() if hasattr(start_time, "isoformat") else str(start_time)
        end_iso = end_time.isoformat() if hasattr(end_time, "isoformat") else str(end_time)
    except Exception:
        raise ValueError("start_time and end_time must be datetime objects or ISO strings")

    service = get_calendar_service()

    event_body = {
        "summary": summary,
        "description": description or "",
        "start": {"dateTime": start_iso, "timeZone": "Asia/Kolkata"},
        "end": {"dateTime": end_iso, "timeZone": "Asia/Kolkata"},
        "organizer": {"email": DELEGATED_USER},
        "attendees": [{"email": attendee_email}] if attendee_email else []
        
    }

    try:
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event_body,
            sendUpdates=send_updates  
        ).execute()

        return {
            "status": "created",
            "eventId": created_event.get("id"),
            "htmlLink": created_event.get("htmlLink"),
            "raw": created_event
        }

    except HttpError as e:
        raise RuntimeError(f"Google API error: {e.resp.status} - {getattr(e, '_get_reason', lambda: e)()}")
    except Exception as e:
        raise RuntimeError(f"Failed to create event: {e}")




# start = datetime(2025, 11, 13, 10, 0).astimezone()   
# end   = datetime(2025, 11, 13, 10, 30).astimezone()

# res = create_event_without_meet(
#     attendee_email="usm@thescooter.ai",
#     start_time=start,
#     end_time=end,
#     summary="video interview",
#     description="Remainder to take video interbiew",
#     send_updates="all"
# )

# print(res)