import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import os
from dotenv import load_dotenv
def send_remind_later_email(candidate_name: str, remind_at_str: str, candidate_email: str, candidate_phone: str):
    """
    Sends a transactional email to the support team when a candidate
    requests to be reminded later for their audio round.
    """
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

    # Build the email
    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[
            {"email": "vishnu@thescooter.ai", "name": "Vishnu"},
            {"email": "ketaki@thescooter.ai", "name": "Ketaki"}
        ],
        sender={"email": "internal@thescooter.ai", "name": "scooter internal"},
        subject="Reminder Notification - Candidate Audio Round",
        html_content=f"""
        <html>
        <body>
        
            <h1>Reminder Notification</h1>
            <p>Dear Support Team,</p>
            <p>Candidate <strong>{candidate_name}</strong> has requested to be reminded at 
            <strong>{remind_at_str}</strong> to take their audio round.</p>
            <p>Candidate Email: <strong>{candidate_email}</strong></p>
            <p>Candidate Phone: <strong>{candidate_phone}</strong></p>
        </body>
        </html>
        """
    )

    try:
        response = api_instance.send_transac_email(send_smtp_email)
       
    except ApiException as e:
        print("Exception when sending email:", e)
       
async def send_password_reset_email(user_email: str, otp:str, user_name: str):
    """
    Sends a password reset email to the user.
    """
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")

    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[
            {"email": user_email, "name": user_name}, 
        ],
        sender={"email": "no-reply@thescooter.ai", "name": "Scooter"},
        subject="Your Secret OTP Has Arrived!",
        html_content=f"""
<html>
  <body style="font-family: Arial, sans-serif; background-color: #ffffff; padding: 20px; margin: 0;">
    <div style="max-width: 600px; margin: auto; border: 1px solid #e0e0e0; border-radius: 8px; padding: 25px;">
      <h2 style="color: #333;">Hellooo there!</h2>
      <p style="font-size: 16px; color: #555;">
        We just sent a super secret code to your inbox (okay, it's not that secret, but it's yours and only yours!🎉)
      </p>
      <p style="font-size: 18px; color: #222; font-weight: bold; margin-top: 20px;">
        Your One-Time Password:
      </p>
      <h1 style="color: #0078D7; font-size: 32px; margin: 10px 0;">{otp}</h1>
      <p style="font-size: 16px; color: #555;">
        Valid for the next <strong>10 minutes</strong> ⏰.
      </p>
      <hr style="margin: 25px 0; border: 0; border-top: 1px solid #e0e0e0;" />
      <h3 style="color: #333;">What now?</h3>
      <ol style="font-size: 16px; color: #555; line-height: 1.6;">
        <li>Head back to the password reset page</li>
        <li>Enter this code</li>
        <li>Create a new, memorable password (but not "password123", please!🙈)</li>
      </ol>
      <p style="font-size: 16px; color: #555; margin-top: 30px;">
        Cheers,<br />
        <strong>Team Scooter</strong>
      </p>
      <p style="font-size: 13px; color: #777; margin-top: 20px;">
        PS: This is an automated email. We'd love to chat, but our robots are a bit shy about replying. 
        For real human conversation, reach us at <a href="mailto:support@thescooter.ai" style="color: #0078D7;">support@thescooter.ai</a>.
      </p>
    </div>
  </body>
</html>
"""
    )

    try:
        response = api_instance.send_transac_email(send_smtp_email)
       
    except ApiException as e:
        print("Exception when sending email:", e)
       

async def notify_admins_for_new_lead(name: str, companyEmail: str, phone: str, companyName: str, designation: str,linkedin: str,query: str):
    """
    Sends a notification email to admins when a new lead is created via Contact Us form.
    """
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv("BREVO_API_KEY")
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

    # Build the email
    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=[
            {"email": "kartik@thescooter.ai", "name": "Karthik"},
            #{"email": "ketaki@thescooter.ai", "name": "Ketaki"}
        ],
        sender={"email": "internal@thescooter.ai", "name": "scooter internal"},
        subject="Lead Alert⚠️ - New Contact Us form Submission",
        html_content=f"""
        <html>
  <body style="font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 20px; margin: 0;">
    <div style="max-width: 650px; margin: auto; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 25px;">
      
      <h2 style="color: #333; border-bottom: 1px solid #e0e0e0; padding-bottom: 10px;">
          New Contact Us Submission
      </h2>
      
      <p style="font-size: 16px; color: #555;">
        A new contact form has been submitted. Here are the details:
      </p>
      
      <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
        <tr>
          <td style="padding: 8px; font-weight: bold; color: #333; width: 150px;">Name:</td>
          <td style="padding: 8px; color: #555;">{name}</td>
        </tr>
        <tr>
          <td style="padding: 8px; font-weight: bold; color: #333;">Email:</td>
          <td style="padding: 8px; color: #555;">{companyEmail}</td>
        </tr>
        <tr>
          <td style="padding: 8px; font-weight: bold; color: #333;">Phone:</td>
          <td style="padding: 8px; color: #555;">{phone}</td>
        </tr>
        <tr>
          <td style="padding: 8px; font-weight: bold; color: #333;">Company:</td>
          <td style="padding: 8px; color: #555;">{companyName}</td>
        </tr>
        <tr>
          <td style="padding: 8px; font-weight: bold; color: #333;">Designation:</td>
          <td style="padding: 8px; color: #555;">{designation}</td>
        </tr>
        <tr>
          <td style="padding: 8px; font-weight: bold; color: #333;">LinkedIn:</td>
          <td style="padding: 8px; color: #555;">{linkedin}</td>
        </tr>
        <tr>
          <td style="padding: 8px; font-weight: bold; color: #333; vertical-align: top;">Query:</td>
          <td style="padding: 8px; color: #555;">{query}</td>
        </tr>
      </table>
      
      <p style="font-size: 14px; color: #777; margin-top: 25px;">
        This is an automated notification. Please follow up promptly with the user.
      </p>
      
    </div>
  </body>
</html>

        """
    )

    try:
        response = api_instance.send_transac_email(send_smtp_email)
       
    except ApiException as e:
        print("Exception when sending email:", e)