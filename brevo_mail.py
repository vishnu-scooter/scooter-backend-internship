# import sib_api_v3_sdk
# from sib_api_v3_sdk.rest import ApiException

# # Configure API key authorization
# configuration = sib_api_v3_sdk.Configuration()
# configuration.api_key['api-key'] = "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"

# # Create an instance of the API class
# api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

# # Build the email
# send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
#     to=[
#         {"email": "vishnu@thescooter.ai", "name": "Vishnu"},
#         {"email": "ketaki@thescooter.ai", "name": "Ketaki"}
#     ],
#     sender={"email": "no-reply@thescooter.ai", "name": "test1"},
#     subject="test notification for audio remind later",
#     html_content="<html><body><h1>below isn the sample email when a candidate is to be reminded for audio later</h1><p>dear support team candidate {name} has requested them to be reminded at {datetime} to take thei audio round.</p></body></html>"
# )

# try:
#     # Send email
#     response = api_instance.send_transac_email(send_smtp_email)
#     print("Email sent successfully:", response)
# except ApiException as e:
#     print("Exception when sending email:", e)



import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

def send_remind_later_email(candidate_name: str, remind_at_str: str, candidate_email: str, candidate_phone: str):
    """
    Sends a transactional email to the support team when a candidate
    requests to be reminded later for their audio round.
    """
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = "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"

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
       
