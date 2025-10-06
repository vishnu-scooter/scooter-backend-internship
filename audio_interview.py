from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from vapi import Vapi
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="AI Interview Caller API", version="1.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vapi client
VAPI_TOKEN = "7b38bd97-6291-453e-91f5-0301f82efd4c"
PHONE_NUMBER_ID = "c6f0577d-4775-4b0f-87ba-f70d90e4670c"

client = Vapi(token=VAPI_TOKEN)

# Pydantic models
class InterviewCallRequest(BaseModel):
    phone_number: str = Field(..., description="Phone number to call (e.g., +916383786120)")
    resume_text: str = Field(..., description="Complete resume text of the candidate")
    system_prompt: str = Field(..., description="Custom system prompt for the interview")
    candidate_name: Optional[str] = Field(None, description="Candidate's name (optional)")
    position: Optional[str] = Field("Sales Representative", description="Position being interviewed for")

class InterviewCallResponse(BaseModel):
    call_id: str
    status: str
    message: str
    candidate_phone: str
    timestamp: str

# Main interview function
@app.post("/api/start-interview-call", response_model=InterviewCallResponse)
async def start_interview_call(request: InterviewCallRequest):
    """
    Start an automated interview call for a sales role based on resume text
    
    This endpoint creates a dynamic VAPI assistant that conducts a personalized
    interview based on the candidate's resume and custom system prompt.
    """
    try:
        logger.info(f"Starting interview call for {request.phone_number}")
        
        # Create dynamic assistant configuration
        assistant_config = create_interview_assistant(
            resume_text=request.resume_text,
            system_prompt=request.system_prompt,
            candidate_name=request.candidate_name,
            position=request.position
        )
        
        # Create the call with VAPI
        call = client.calls.create(
            phone_number_id=PHONE_NUMBER_ID,
            customer={
                "number": request.phone_number,
                "name": request.candidate_name or "Candidate"
            },
            **assistant_config
        )
        
        logger.info(f"Call created successfully with ID: {call.id}")
        
        return InterviewCallResponse(
            call_id=call.id,
            status="success",
            message="Interview call initiated successfully",
            candidate_phone=request.phone_number,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating interview call: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start interview call: {str(e)}"
        )

def create_interview_assistant(
    resume_text: str, 
    system_prompt: str, 
    candidate_name: Optional[str] = None,
    position: str = "Sales Representative"
) -> Dict[str, Any]:
    """
    Create a dynamic assistant configuration for conducting sales interviews
    """
    
    # Build comprehensive system message
    base_system_message = f"""
    {system_prompt}
    
    CANDIDATE INFORMATION:
    Name: {candidate_name or 'Candidate'}
    Position: {position}
    
    RESUME DETAILS:
    {resume_text}
    
    INTERVIEW INSTRUCTIONS:
    1. Conduct a professional sales role interview
    2. Ask relevant questions based on the candidate's resume
    3. Evaluate their sales experience, skills, and communication
    4. Keep the interview focused and engaging (15-20 minutes)
    5. Be encouraging but thorough in your assessment
    6. End with next steps information
    
    Start by greeting the candidate warmly and confirming their availability for the interview.
    """
    
    # Create first message based on candidate info
    first_message = f"Hello{f' {candidate_name}' if candidate_name else ''}! This is the AI interviewer calling about the {position} position. Do you have about 15-20 minutes to discuss your background and experience?"
    
    return {
        "assistant": {
            "model": {
                "provider": "openai",
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": base_system_message
                    }
                ],
                "temperature": 0.7,
                "maxTokens": 500
            },
            "voice": {
                "provider": "11labs", 
                "voiceId": "shimmer"
            },
            "firstMessage": first_message,
            "recordingEnabled": True,
            "endCallMessage": f"Thank you for your time{f' {candidate_name}' if candidate_name else ''}. We'll be in touch with next steps soon. Have a great day!",
            "endCallPhrases": ["thank you for the interview", "that concludes our interview", "we'll be in touch"],
            "interruptionsEnabled": True,
            "responseDelaySeconds": 1,
            "llmRequestDelaySeconds": 0.1,
            "numWordsToInterruptAssistant": 2,
            "maxDurationSeconds": 1800,  # 30 minutes max
            "backgroundSound": "office"
        }
    }

# Additional endpoints for managing interviews

@app.get("/api/call-status/{call_id}")
async def get_call_status(call_id: str):
    """
    Get the status of an ongoing or completed interview call
    """
    try:
        call = client.calls.get(id=call_id)
        return {
            "call_id": call_id,
            "status": call.status,
            "duration": getattr(call, 'duration', None),
            "cost": getattr(call, 'cost', None)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Call not found: {str(e)}")

@app.post("/api/end-call/{call_id}")
async def end_interview_call(call_id: str):
    """
    Manually end an ongoing interview call
    """
    try:
        # End the call
        result = client.calls.update(
            id=call_id,
            status="ended"
        )
        return {
            "call_id": call_id,
            "status": "ended",
            "message": "Interview call ended successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to end call: {str(e)}")

# Batch interview endpoint
class BatchInterviewRequest(BaseModel):
    candidates: list[InterviewCallRequest] = Field(..., description="List of candidates to interview")
    delay_between_calls: int = Field(300, description="Delay between calls in seconds (default: 5 minutes)")

@app.post("/api/start-batch-interviews")
async def start_batch_interviews(request: BatchInterviewRequest):
    """
    Start multiple interview calls in sequence with delays
    """
    import asyncio
    
    results = []
    
    for i, candidate in enumerate(request.candidates):
        try:
            # Add delay between calls (except for the first one)
            if i > 0:
                await asyncio.sleep(request.delay_between_calls)
            
            result = await start_interview_call(candidate)
            results.append({
                "candidate_phone": candidate.phone_number,
                "result": "success",
                "call_id": result.call_id
            })
            
        except Exception as e:
            results.append({
                "candidate_phone": candidate.phone_number,
                "result": "failed",
                "error": str(e)
            })
    
    return {
        "total_candidates": len(request.candidates),
        "results": results,
        "batch_started": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Check if the service is running and VAPI client is accessible
    """
    try:
        # Test VAPI connection
        phone_numbers = client.phone_numbers.list()
        return {
            "status": "healthy",
            "vapi_connected": True,
            "phone_numbers_available": len(phone_numbers) > 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "vapi_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
