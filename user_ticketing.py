from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from datetime import datetime, timezone
import os, logging, random, string
from typing import Optional
import requests
import uvicorn
from dotenv import load_dotenv

load_dotenv()
# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    filename="user-ticketing.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
app = FastAPI()

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URL")
client = AsyncIOMotorClient(MONGO_URI)
db = client["scooter_ai_db"]
collection = db["user-ticket"]
# Azure settings (replace with your actual settings or use env variables)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_SCREENSHOT_CONTAINER_NAME = os.getenv("AZURE_STORAGE_SCREENSHOT_CONTAINER_NAME", "user-screenshots")
# Reference number generator
def generate_short_reference(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# Mailgun email sender
def send_support_conformation_email(to_email: str, reference_number: str, name: str):
    try:
        response = requests.post(
            "https://api.mailgun.net/v3/thescooter.ai/messages",
            auth=("api", os.getenv('mailgun_api', '6fa54e2c5c1970f59575773a5e8dafdc-812b35f5-c046da19')),
            data={
                "from": "ScooterAI Support <support@thescooter.ai>",
                "to": to_email,
                "subject": "Support Ticket Created",
                "text": f"""Dear {name},

Thank you for contacting Scooter AI Support.

We’ve received your request and have created a support ticket with the reference number: {reference_number}. Our team is actively reviewing your issue and working to resolve it as quickly as possible.

If you need to follow up, please mention this reference number in any future communication to help us assist you more efficiently.

We sincerely apologize for any inconvenience caused and appreciate your patience during this time. We’ll notify you as soon as your query has been resolved.

Best regards,  
Scooter AI Support Team"""
            }
        )
        logger.info(f"Email sent to {to_email} with status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

def notify_developer_of_new_ticket(ticket: dict):
    try:
        response = requests.post(
            "https://api.mailgun.net/v3/thescooter.ai/messages",
            auth=("api", os.getenv('mailgun_api', '6fa54e2c5c1970f59575773a5e8dafdc-812b35f5-c046da19')),
            data={
                "from": "ScooterAI automation <support@thescooter.ai>",
                "to": "vishnu@thescooter.ai,usman@thescooter.ai,priyanshu@thescooter.ai",  # Replace with your actual dev/support email
                "subject": f"[New Support Ticket] {ticket.get('reference_number', 'N/A')}",
                "text": f""" Dear developer New support ticket has been received.

Reference Number: {ticket.get('reference_number')}
Name: {ticket.get('name')}
Email: {ticket.get('email')}
Phone: {ticket.get('phonenumber')}
Description:
{ticket.get('description')}
screenshot_url: {ticket.get('screenshot_url', 'Not provided')}
Created At: {ticket.get('created_at')}

"""
            }
        )
        logger.info(f"Developer notification sent for ticket {ticket.get('reference_number')}, status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to notify developer for ticket {ticket.get('reference_number')}: {e}")
        return False

async def upload_to_blob_storage_screenshot(file: UploadFile, reference_number: str) -> str:
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        extension = os.path.splitext(file.filename)[1]
        blob_name = f"screenshot-{reference_number}-{timestamp}{extension}"

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_SCREENSHOT_CONTAINER_NAME)

        try:
            await container_client.create_container()
        except ResourceExistsError:
            pass

        blob_client = container_client.get_blob_client(blob_name)
        content = await file.read()
        await blob_client.upload_blob(content, overwrite=True)
        return blob_client.url
    except Exception as e:
        logger.error(f"Screenshot upload failed: {e}")
        raise HTTPException(status_code=500, detail="Screenshot upload failed.")

@app.post("/submit-ticket/")
async def submit_ticket(
    name: str = Form(...),
    email: EmailStr = Form(...),
    phonenumber: str = Form(...),
    description: str = Form(...),
    screenshot: Optional[UploadFile] = File(None)
):
    try:
        reference_number = generate_short_reference()
        created_at = datetime.now(timezone.utc).isoformat()

        ticket_dict = {
            "name": name,
            "email": email,
            "phonenumber": phonenumber,
            "description": description,
            "reference_number": reference_number,
            "created_at": created_at
        }

        # Upload screenshot if provided
        if screenshot:
            screenshot_url = await upload_to_blob_storage_screenshot(screenshot, reference_number)
            ticket_dict["screenshot_url"] = screenshot_url

        await collection.insert_one(ticket_dict)
        notify_developer_of_new_ticket(ticket_dict)
        send_support_conformation_email(email, reference_number, name)
        # You can trigger your Mailgun logic here using `email` and `reference_number`

        return {
            "message": "Issue submitted successfully. Our team will contact you shortly.",
            "reference_number": reference_number
        }
    except Exception as e:
        logger.error(f"Failed to submit ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Start app
if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8002)
