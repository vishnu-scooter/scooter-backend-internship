from fastapi import HTTPException, UploadFile, File, Form
from azure.storage.blob.aio import BlobServiceClient
from azure.core.exceptions import ResourceExistsError
from datetime import datetime, timezone
import os, logging, random, string
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
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



# Azure settings (replace with your actual settings or use env variables)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_SCREENSHOT_CONTAINER_NAME = os.getenv("AZURE_STORAGE_SCREENSHOT_CONTAINER_NAME", "user-screenshots")
AZURE_STORAGE_COMPANYLOGO_CONTAINER_NAME = os.getenv("AZURE_STORAGE_COMPANYLOGO_CONTAINER_NAME", "company-logos")
# Reference number generator
def generate_short_reference(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# Support Confirmation Email
# Support Confirmation Email
async def send_support_conformation_email(to_email: str, reference_number: str, name: str):
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

    subject = "Support Ticket Created"
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #ffffff; padding: 20px;">
        <div style="max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 25px;">
          <h2 style="color: #333;">Dear {name},</h2>
          <p style="font-size: 16px; color: #555;">
            Thank you for contacting Scooter AI Support.
          </p>
          <p style="font-size: 16px; color: #555;">
            We’ve received your request and have created a support ticket with the reference number: 
          </p>
          <h3 style="color: #0078D7;">{reference_number}</h3>
          <p style="font-size: 16px; color: #555;">
            Our team is actively reviewing your issue and working to resolve it as quickly as possible.
          </p>
          <p style="font-size: 16px; color: #555;">
            If you need to follow up, please mention this reference number in any future communication to help us assist you more efficiently.
          </p>
          <p style="font-size: 16px; color: #555;">
            We sincerely apologize for any inconvenience caused and appreciate your patience during this time. We’ll notify you as soon as your query has been resolved.
          </p>
          <p style="font-size: 16px; color: #555; margin-top: 30px;">
            Best regards,<br />
            <strong>Scooter AI Support Team</strong>
          </p>
        </div>
      </body>
    </html>
    """

    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": to_email, "name": name}],
        sender={"email": "support@thescooter.ai", "name": "ScooterAI Support"},
        subject=subject,
        html_content=html_content,
    )

    try:
        response = api_instance.send_transac_email(send_smtp_email)
        logger.info(f"Email sent to {to_email} via Brevo, response: {response}")
        return True
    except ApiException as e:
        logger.error(f"Failed to send Brevo email to {to_email}: {e}")
        return False


# Developer Notification Email
async def notify_developer_of_new_ticket(ticket: dict):
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

    subject = f"[New Support Ticket] {ticket.get('reference_number', 'N/A')}"
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #ffffff; padding: 20px;">
        <div style="max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 25px;">
          <h2 style="color: #333;">Dear developer,</h2>
          <p style="font-size: 16px; color: #555;">New support ticket has been received. Details below:</p>
          <ul style="font-size: 16px; color: #555; line-height: 1.6;">
            <li><strong>Reference Number:</strong> {ticket.get('reference_number')}</li>
            <li><strong>Name:</strong> {ticket.get('name')}</li>
            <li><strong>Email:</strong> {ticket.get('email')}</li>
            <li><strong>Phone:</strong> {ticket.get('phonenumber')}</li>
            <li><strong>Description:</strong> {ticket.get('description')}</li>
            <li><strong>Screenshot URL:</strong> {ticket.get('screenshot_url', 'Not provided')}</li>
            <li><strong>Created At:</strong> {ticket.get('created_at')}</li>
          </ul>
        </div>
      </body>
    </html>
    """

    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[
            {"email": "vishnu@thescooter.ai"},
            {"email": "usman@thescooter.ai"},
            {"email": "priyanshu@thescooter.ai"},
            {"email": "ketaki@thescooter.ai"},
        ],
        sender={"email": "support@thescooter.ai", "name": "ScooterAI Automation"},
        subject=subject,
        html_content=html_content,
    )

    try:
        response = api_instance.send_transac_email(send_smtp_email)
        logger.info(f"Developer notification sent for ticket {ticket.get('reference_number')}, response: {response}")
        return True
    except ApiException as e:
        logger.error(f"Failed to notify developer via Brevo for ticket {ticket.get('reference_number')}: {e}")
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


async def upload_to_blob_storage_company_logo(file: UploadFile, reference_number: str) -> str:
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        extension = os.path.splitext(file.filename)[1]
        blob_name = f"logo-{reference_number}-{timestamp}{extension}"

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_COMPANYLOGO_CONTAINER_NAME)

        try:
            await container_client.create_container()
        except ResourceExistsError:
            pass

        blob_client = container_client.get_blob_client(blob_name)
        content = await file.read()
        await blob_client.upload_blob(content, overwrite=True)
        return blob_client.url
    except Exception as e:
        logger.error(f"logo upload failed: {e}")
        raise HTTPException(status_code=500, detail="logo upload failed.")