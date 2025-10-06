from vapi import Vapi

client = Vapi(
    token="7b38bd97-6291-453e-91f5-0301f82efd4c",
)

# Get an assistant (optional, if you want to verify it's accessible)
assistant = client.assistants.get(
    id="c668b4a6-3ebf-47ed-be1b-faa8240f33dd",
)

# Create an inbound call
call = client.calls.create(
    assistant_id="c668b4a6-3ebf-47ed-be1b-faa8240f33dd",  # existing assistant
    phone_number_id="c6f0577d-4775-4b0f-87ba-f70d90e4670c",               # your registered number in Vapi
    customer={                                            # transient customer
        "name": "vishnu varadhan",
        "number": "+916383786120"                     # replace with the caller’s number
    }
)



print(call)