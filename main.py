from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Form
from fastapi import Query
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Union,Literal
import os
from dotenv import load_dotenv
from pydantic_models import Item, ResumeData
import aiohttp
import json
from pypdf import PdfReader
import io
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import re
import random
import asyncio
import uuid
import time
import csv
import io
from resume_ocr import extract_text_with_ocr
from bson import ObjectId
from fastapi.responses import JSONResponse, StreamingResponse
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError
import base64
from azure.storage.blob import BlobBlock
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import ContentSettings
import bcrypt
from question_banks import (
    GRIT_GUIDING_QUESTIONS, 
    LEARNING_AGILITY_GUIDING_QUESTIONS, 
    LEARNING_AGILITY_PROBING_QUESTIONS, 
    PROBING_QUESTION_TEMPLATES,
    COMMERCIAL_ACUMEN_GUIDING_QUESTIONS,
    COMMERCIAL_ACUMEN_PROBING_QUESTIONS
)
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import string
from create_jd import call_openai_for_jd
from user_ticketing import send_support_conformation_email, notify_developer_of_new_ticket, generate_short_reference,upload_to_blob_storage_screenshot
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    filename="dev.log",  # Name of the log file
    filemode="a",        # Append to the file (use "w" to overwrite each time)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Log format
)

logger = logging.getLogger(__name__)
logger.info("application startup good")

# Load environment variables
load_dotenv()

app = FastAPI(title="FastAPI MongoDB App")

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL")
client = AsyncIOMotorClient(MONGODB_URL)
db = client["scooter_ai_db"]

class Settings:
    MONGO_URI: str = os.getenv("MONGO_URI")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME")
    AZURE_OPENAI_KEY: str =os.getenv("AZURE_OPENAI_KEY", "  ")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", " ")
    AZURE_DEPLOYMENT_NAME: str = os.getenv("AZURE_DEPLOYMENT_NAME", " ")
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", " ")
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_VIDEO_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_VIDEO_STORAGE_CONNECTION_STRING", " ")
    AZURE_STORAGE_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_CONTAINER_NAME", " ")
    AZURE_VIDEO_STORAGE_CONTAINER_NAME: str = os.getenv("AZURE_VIDEO_STORAGE_CONTAINER_NAME", " ")
    AZURE_STORAGE_AUDIO_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_AUDIO_CONTAINER_NAME", " ")
    AZURE_STORAGE_RESUME_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_RESUME_CONTAINER_NAME", " ")
    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY")
    FROM_EMAIL: str = os.getenv("FROM_EMAIL")

settings = Settings()
# Azure OpenAI API configuration
AZURE_OPENAI_URL = f"{settings.AZURE_OPENAI_ENDPOINT}/openai/deployments/{settings.AZURE_DEPLOYMENT_NAME}/chat/completions?api-version={settings.AZURE_API_VERSION}"

# Headers for Azure OpenAI API
AZURE_HEADERS = {
    "Content-Type": "application/json",
    "api-key": settings.AZURE_OPENAI_KEY
}

def serialize_document(doc):
    if not doc:
        return doc
    serialized = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            serialized[k] = str(v)
        elif isinstance(v, datetime):
            serialized[k] = v.isoformat()
        else:
            serialized[k] = v
    return serialized

def is_likely_resume(text: str) -> bool:
    """
    Perform multiple checks to determine if the document is likely a resume
    """
    if len(text) < 500:
        logger.warning("Resume is too short")
        return False
    
    resume_keywords = [
        'curriculum vitae', 'cv', 'resume', 'contact information',
        'work experience', 'professional summary', 'skills',
        'education', 'professional experience', 'career objective',
        'linkedin', 'email', 'phone', 'address'
    ]
    
    keyword_matches = sum(
        1 for keyword in resume_keywords
        if keyword.lower() in text.lower()
    )
    
    if keyword_matches < 3:
        logger.warning(f"Insufficient resume keywords. Matches: {keyword_matches}")
        return False
    
    # pattern_checks = [
    #     bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
    #     bool(re.search(r'\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b', text)),
    #     bool(re.search(r'\b(experience|worked|employed|position|role)\b', text, re.IGNORECASE))
    # ]
    
    # if sum(pattern_checks) < 2:
    #     logger.warning("Failed resume pattern checks")
    #     return False
    
    logger.info("Resume pattern check successful")
    return True

async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    logger.info(f"Starting PDF text extraction for file: {pdf_file.filename}")
    try:
        content = await pdf_file.read()
        pdf_reader = PdfReader(io.BytesIO(content))
        
        if len(pdf_reader.pages) > 10:
            logger.warning("Resume exceeds typical page limit")
            raise HTTPException(status_code=400, detail="Resume exceeds typical page limit")
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            text += page_text
            logger.debug(f"Extracted text from page {page_num}")
        
        
        logger.info(f"Successfully extracted text from {len(pdf_reader.pages)} pages")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

async def parse_resume_with_azure(text: str) -> dict:
    logger.info("Starting resume parsing with Azure OpenAI")
    prompt = f"""Please analyze the following resume and extract the information in the exact JSON structure shown below. If any information is not available, use empty strings ("") for text fields, empty arrays ([]) for lists, and false for booleans. Do not include any additional text or explanation, only return the JSON object.

IMPORTANT: You MUST extract company history from the resume. Look for sections like "Work Experience", "Professional Experience", "Employment History", etc. For each role, extract:
- Company name
- Position/title
- Start date
- End date (use 'Present' if current)
- Calculate duration in months
- Set is_current to true if it's the current role

Date Handling Rules:
1. If only month and year are mentioned (e.g., "Jan 2020" or "January 2020"):
   - Start date: Use first day of that month (YYYY-01-01)
   - End date: Use last day of that month (YYYY-12-31)
2. If only year is mentioned (e.g., "2020"):
   - Start date: Use April 1st of that year (YYYY-04-01)
   - End date: Use March 31st of next year (YYYY+1-03-31)
3. If no date is specified:
   - Use empty string ("") for both start_date and end_date
4. For current roles:
   - End date should be "Present"
   - Set is_current to true
5. All dates should be in YYYY-MM-DD format

Example date conversions:
- "Jan 2020" → start_date: "2020-01-01", end_date: "2020-12-31"
- "2020" → start_date: "2020-04-01", end_date: "2021-03-31"
- "Present" → end_date: "Present", is_current: true
- "Jan 2020 - Present" → start_date: "2020-01-01", end_date: "Present", is_current: true

{{
    "basic_information": {{
        "full_name": "string",
        "current_location": "string",
        "open_to_relocation": boolean,
        "phone_number": "string",
        "linkedin_url": "string",
        "email": "string",
        "specific_phone_number": "string",
        "notice_period": "string",
        "current_ctc": {{
            "currencyType": "string",
            "value": number
        }},
        "expected_ctc": {{
            "currencyType": "string",
            "value": number
        }}
    }},
    "career_overview": {{
        "total_years_experience": float,
        "years_sales_experience": float,
        "average_tenure_per_role": float,
        "employment_gaps": {{
            "has_gaps": boolean,
            "duration": "string"
        }},
        "promotion_history": boolean,
        "company_history": [
            {{
                "company_name": "string",
                "position": "string",
                "start_date": "string (YYYY-MM-DD)",
                "end_date": "string (YYYY-MM-DD) or 'Present'",
                "duration_months": integer,
                "is_current": boolean
            }}
        ]
    }},
    "sales_context": {{
        "sales_type": "string",
        "sales_motion": "string",
        "industries_sold_into": ["string"],
        "regions_sold_into": ["string"],
        "buyer_personas": ["string"]
    }},
    "role_process_exposure": {{
        "sales_role_type": "string",
        "position_level": "string",
        "sales_stages_owned": ["string"],
        "average_deal_size": "string",
        "sales_cycle_length": "string",
        "own_quota": boolean,
        "quota_ownership": ["string"],
        "quota_attainment": "string"
    }},
    "tools_platforms": {{
        "crm_tools": ["string"],
        "sales_tools": ["string"]
    }}
}}

Resume text:
{text}

Remember to:
1. Return ONLY the JSON object, no additional text
2. Use empty strings ("") for missing text fields
3. Use empty arrays ([]) for missing list fields
4. Use false for missing boolean fields
5. Use 0 for missing numeric fields
6. Follow the exact structure shown above
7. For company_history:
   - Extract ALL companies and positions in chronological order
   - Calculate duration_months based on start_date and end_date
   - Set is_current to true if end_date is 'Present'
   - Use YYYY-MM-DD format for dates
   - Sort companies by start_date in descending order (most recent first)
   - Follow the date handling rules specified above
   - If no company history is found, include an empty array []
8. For dates:
   - Follow the date handling rules specified above
   - Use YYYY-MM-DD format
   - For current role, use 'Present' as end_date"""

    try:
        async with aiohttp.ClientSession() as session:
            logger.debug("Sending request to Azure OpenAI API")
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a professional resume analyzer. Extract information in the exact JSON format requested, with special attention to company history and dates. You MUST extract company history from the resume. If no company history is found, include an empty array for company_history. For any missing text fields, use empty strings (''). For any missing list fields, use empty arrays ([]). For any missing boolean fields, use false. For any missing numeric fields, use 0."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}
                }
            ) as response:
                if response.status != 200:
                    error_msg = f"Azure OpenAI API returned status code {response.status}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)
                
                result = await response.json()
                logger.debug("Received response from Azure OpenAI API")
                
                try:
                    if "choices" not in result or not result["choices"]:
                        error_msg = "Invalid response format from Azure OpenAI API"
                        logger.error(error_msg)
                        raise HTTPException(status_code=500, detail=error_msg)

                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"Raw API response content: {content[:200]}...")  # Log first 200 chars for debugging
                    
                    if not content or not content.strip():
                        error_msg = "Empty response from Azure OpenAI API"
                        logger.error(error_msg)
                        raise HTTPException(status_code=500, detail=error_msg)

                    try:
                        parsed_data = json.loads(content)
                        
                        # Validate required fields
                        required_sections = [
                            "basic_information",
                            "career_overview",
                            "sales_context",
                            "role_process_exposure",
                            "tools_platforms"
                        ]
                        
                        missing_sections = [section for section in required_sections if section not in parsed_data]
                        if missing_sections:
                            error_msg = f"Missing required sections in response: {', '.join(missing_sections)}"
                            logger.error(error_msg)
                            raise HTTPException(status_code=500, detail=error_msg)
                        
                        # Ensure all required fields exist with default values
                        default_values = {
                            "basic_information": {
                                "full_name": "",
                                "current_location": "",
                                "open_to_relocation": False,
                                "phone_number": "",
                                "linkedin_url": "",
                                "email": "",
                                "specific_phone_number": "",
                                "notice_period": "",
                                "current_ctc": {"currencyType": "", "value": 0},
                                "expected_ctc": {"currencyType": "", "value": 0}
                            },
                            "career_overview": {
                                "total_years_experience": 0.0,
                                "years_sales_experience": 0.0,
                                "average_tenure_per_role": 0.0,
                                "employment_gaps": {
                                    "has_gaps": False,
                                    "duration": ""
                                },
                                "promotion_history": False,
                                "company_history": []
                            },
                            "sales_context": {
                                "sales_type": "",
                                "sales_motion": "",
                                "industries_sold_into": [],
                                "regions_sold_into": [],
                                "buyer_personas": []
                            },
                            "role_process_exposure": {
                                "sales_role_type": "",
                                "position_level": "",
                                "sales_stages_owned": [],
                                "average_deal_size": "",
                                "sales_cycle_length": "",
                                "own_quota": False,
                                "quota_ownership": [],
                                "quota_attainment": ""
                            },
                            "tools_platforms": {
                                "crm_tools": [],
                                "sales_tools": []
                            }
                        }
                        
                        # Apply default values for missing fields
                        for section, defaults in default_values.items():
                            if section not in parsed_data:
                                parsed_data[section] = defaults
                            else:
                                for field, default_value in defaults.items():
                                    if field not in parsed_data[section]:
                                        parsed_data[section][field] = default_value
                        
                        # Process company history
                        company_history = parsed_data["career_overview"]["company_history"]
                        
                        # Sort companies by start date (most recent first)
                        company_history.sort(key=lambda x: x["start_date"], reverse=True)
                        
                        # Calculate total months and average tenure
                        total_months = sum(role["duration_months"] for role in company_history)
                        if company_history:
                            parsed_data["career_overview"]["average_tenure_per_role"] = round(total_months / len(company_history) / 12, 1)
                            parsed_data["career_overview"]["total_years_experience"] = round(total_months / 12, 1)
                        
                        # Update the sorted company history
                        parsed_data["career_overview"]["company_history"] = company_history
                        
                        logger.info(f"Processed company history with {len(company_history)} companies")
                        logger.info(f"Total experience: {parsed_data['career_overview']['total_years_experience']} years")
                        logger.info(f"Average tenure: {parsed_data['career_overview']['average_tenure_per_role']} years")
                        
                        logger.info("Successfully parsed and validated resume data")
                        return parsed_data
                    except json.JSONDecodeError as e:
                        error_msg = f"Invalid JSON response from Azure OpenAI API: {str(e)}\nContent: {content[:200]}..."
                        logger.error(error_msg)
                        raise HTTPException(status_code=500, detail=error_msg)
                except (KeyError, IndexError) as e:
                    error_msg = f"Error accessing Azure OpenAI response: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    raise HTTPException(status_code=500, detail=error_msg)
    except aiohttp.ClientError as e:
        error_msg = f"Network error calling Azure OpenAI API: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error calling Azure OpenAI API: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

async def upload_to_blob_storage_resume(file: UploadFile, user_id: str) -> tuple[str, str]:
    """
    Upload a resume file to Azure Blob Storage and return the URL and blob name.
    """
    try:
        # Generate a unique blob name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        blob_name = f"{user_id}-{timestamp}{file_extension}"
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        
        # Get container client
        container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_RESUME_CONTAINER_NAME)
        
        # Create container if it doesn't exist
        try:
            await container_client.create_container()
        except ResourceExistsError:
            pass
        
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Read file content
        content = await file.read()
        
        # Upload the file
        await blob_client.upload_blob(content, overwrite=True)
        
        # Get the URL
        resume_url = blob_client.url
        
        return resume_url, blob_name
    except Exception as e:
        logger.error(f"Error uploading resume to blob storage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {str(e)}")

async def is_valid_resume(file: UploadFile) -> bool:
    """
    Check if the uploaded file is a valid resume by analyzing its content.
    Returns True if it's a valid resume, False otherwise.
    """
    try:
        # Read file content
        content = await file.read()
        if not content:
            return False
            
        # Create a new UploadFile for text extraction
        file_for_text = UploadFile(
            filename=file.filename,
            file=io.BytesIO(content)
        )
            
        # Extract text from PDF
        text = await extract_text_from_pdf(file_for_text)
        if not text:
            return False
            
        # Check if the text contains resume-like content
        # Look for common resume sections and keywords
        resume_keywords = [
            'experience', 'education', 'skills', 'work', 'employment',
            'career', 'professional', 'summary', 'objective', 'achievements',
            'responsibilities', 'projects', 'certifications', 'languages',
            'references', 'contact', 'phone', 'email', 'linkedin'
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in resume_keywords if keyword in text_lower)
        
        # If we find at least 3 resume-related keywords, consider it a valid resume
        return keyword_matches >= 3
    except Exception as e:
        logger.error(f"Error validating resume: {str(e)}")
        return False
    
class UserAccount(BaseModel):
    email: str = Field(..., description="User's email address")
    name: str = Field(..., description="User's full name")
    number: str = Field(..., description="User's phone number")

@app.post("/create-user-account/")
async def create_user_account(user: UserAccount):
    """
    Create a new user account with email, name, and phone number.
    """
    logger.info(f"Creating user account for email: {user.email}")
    collection = db["user_accounts"]
    
    existing_user = await collection.find_one({"email": user.email})
    
    if existing_user:
        logger.warning("User already exists")
        raise HTTPException(status_code=400, detail="User already exists")
    
    user_dict = user.dict()
    user_dict["application_ids"] = []
    
    result = await collection.insert_one(user_dict)
    user_id = str(result.inserted_id)
    
    logger.info(f"User account created with ID: {user_id}")
    return JSONResponse(
        status_code=200,
        content={
            "status": True,
            "message": "User account created successfully",
            "user_id": user_id
        }
    )

class UserAccountLogin(BaseModel):
    email: str 
    #number: str 

@app.post("/user-login/")
async def login(request: UserAccountLogin):
    logger.info(f"Login request for email: {request.email}")
    collection = db["user_accounts"]
    profile_collection = db["resume_profiles"]
    job_collection = db["job_roles"]

    user = await collection.find_one({"email": request.email})

    if not user:
        logger.warning("User not found")
        return JSONResponse(
            status_code=200,
            content={
                "status": False,
                "message": "User not found",
                "data": {}
            }
        )

    logger.info(f"User found: {user['_id']}")

    # Get last application ID
    application_ids = user.get("application_ids", [])
    last_application_id = application_ids[-1] if application_ids else None

    # Get resume profile
    profile_data = await profile_collection.find_one({"_id": ObjectId(last_application_id)}) if last_application_id else None

    # Get job_id from resume profile
    job_id = profile_data.get("job_id") if profile_data else None

    # Get job details
    job = await job_collection.find_one({"_id": ObjectId(job_id)}) if job_id else None

    return JSONResponse(
        status_code=200,
        content={
            "status": True,
            "message": "User found",
            "account_id": str(user["_id"]),
            "data": {
                "email": user["email"],
                "name": user.get("name", ""),
                "phone": user.get("phone", ""),
                "last_application_id": last_application_id,
                "audio_interview_attended": "interview_otp" in profile_data if profile_data else False
            },
            "job_data": {
                "job_id": str(job_id) if job_id else "",
                "job_title": job.get("title", "") if job else "",
                "job_description": job.get("description", "") if job else ""
            }
        }
    )

class CandidateProfileData(BaseModel):
    application_id: Optional[str]
    account_id: Optional[str] 
    basic_information: Optional[Dict[str, Any]] = {}
    application_status: Optional[str] = ""
    #application_status_reason: Optional[str] = ""
    career_overview: Optional[Dict[str, Any]] = {}
    sales_context: Optional[Dict[str, Any]] = {}
    role_process_exposure: Optional[Dict[str, Any]] = {}
    tools_platforms: Optional[Dict[str, Any]] = {}
    resume_url: Optional[str] = None
    video_interview_started: Optional[bool] = False
    audio_interview_attended: Optional[bool] = False

@app.get("/candidate-profile/", response_model=CandidateProfileData)
async def get_candidate_profile(
    account_id: str = Query(..., description="User ID to fetch profile for")
):
    """
    Get candidate profile by account ID.
    """
    logger.info(f"Fetching candidate profile for account ID: {account_id}")
    collection = db["user_accounts"]

    if not ObjectId.is_valid(account_id):
        logger.error("Invalid account ID format")
        raise HTTPException(status_code=400, detail="Invalid account ID format")

    profile = await collection.find_one({"_id": ObjectId(account_id)})

    if not profile:
        logger.warning("Profile not found")
        return JSONResponse(
            status_code=200,
            content={
                "status": False,
                "message": "Profile not found",
                "data": {}
            }
        )

    application_ids = profile.get("application_ids", [])
    data_id = application_ids[-1] if application_ids else None
    profile_data = await db["resume_profiles"].find_one({"_id": ObjectId(data_id)}) if data_id else None

    if not profile_data:
        logger.warning("Profile data not found")
        return JSONResponse(
            status_code=200,
            content={
                "status": False,
                "message": "Profile data not found",
                "data": {}
            }
        )

    candidate = {
        "application_id": str(profile_data["_id"]),
        "account_id": account_id,
        "basic_information": profile_data.get("basic_information", {}),
        "career_overview": profile_data.get("career_overview", {}),
        "interview_status": {
            "audio_interview_passed": profile_data.get("audio_interview", False),
            "video_interview_attended": bool(profile_data.get("video_url")) if profile_data.get("video_url") else False,
            "audio_interview_attended": bool(profile_data.get("audio_url")) if profile_data.get("audio_url") else False,
            "video_interview_url": profile_data.get("video_url") if profile_data.get("video_url") else None,
            "audio_interview_url": profile_data.get("audio_url") if profile_data.get("audio_url") else None,
            "resume_url": profile_data.get("resume_url") if profile_data.get("resume_url") else None
        }
    }
    return JSONResponse(
        status_code=200,
        content={
            "status": True,
            "message": "Candidate profile fetched successfully",
            "data": CandidateProfileData(**candidate).dict()
        }
    )
    #return CandidateProfileData(**candidate)



@app.post("/parse-resume/", response_model=ResumeData)
async def parse_resume(
    file: UploadFile = File(...),
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    candidate_source: Optional[str] = Form(None)
):
    """
    Parse a resume and extract structured information.
    Upload to Azure Blob Storage even if text extraction fails.
    """
    logger.info(f"Received resume parsing request for file: {file.filename}")
    collection = db["resume_profiles"]
    user_accounts = db["user_accounts"]

    # existing_user = await collection.find_one({"email": email})
    # if existing_user:
    #     logger.info(f"User with email {email} already exists")

    #     # Check if key profile sections exist
    #     important_keys = [
    #         "basic_information", "tools_platforms",
    #         "sales_context", "role_process_exposure", "career_overview"
    #     ]

    #     if any(k in existing_user for k in important_keys):
    #         logger.info("Returning existing parsed profile details")
    #         return JSONResponse(
    #             status_code=200,
    #             content={
    #                 "status": True,
    #                 "message": "User profile already exists and properly setup",
    #                 "user_id": str(existing_user["_id"]),
    #                 "data": ResumeData(**existing_user).dict()
    #             }
    #         )
    #     elif existing_user.get("resume_text"):
    #         logger.info("Resume has already been uploaded and parsed")
    #         return JSONResponse(
    #             status_code=200,
    #             content={
    #                 "status": False,
    #                 "user_id": str(existing_user["_id"]),
    #                 "message": "Resume has been parsed and uploaded. Proceed with profile creation or update.",
    #                 "resume_url": existing_user.get("resume_url")
    #             }
    #         )
    #     else:
    #         logger.info("User found, but resume is not parseable")
    #         return JSONResponse(
    #             status_code=200,
    #             content={
    #                 "status": False,
    #                 "user_id": str(existing_user["_id"]),
    #                 "resume_url": existing_user.get("resume_url"),
    #                 "message": "User is present but resume is not parseable"
    #             }
    #         )

    try:
        # Read file content once
        content = await file.read()
        if not content:
            return JSONResponse(status_code=200, content={"message": "File not readable"})

        # Upload to Azure Blob Storage
        resume_url = None
        if file.filename.lower().endswith(".pdf"):
            file_for_blob = UploadFile(
                filename=file.filename,
                file=io.BytesIO(content)
            )
            resume_url, resume_blob_name = await upload_to_blob_storage_resume(file_for_blob, "resume")
            logger.info(f"Resume uploaded to blob: {resume_url}")

        # Try extracting text from the file
        file_for_text = UploadFile(
            filename=file.filename,
            file=io.BytesIO(content)
        )
        file_bytes = await file_for_text.read()
        file_for_text.file.seek(0)
        #text = await extract_text_from_pdf(file_for_text)
        try:
            # Pass BytesIO so PyPDF and OCR both can reuse the same content
            text = await extract_text_from_pdf(UploadFile(filename=file_for_text.filename, file=io.BytesIO(file_bytes)))
            if not text:
                logger.info("PyPDF returned empty text, falling back to Azure OCR")
                text = await extract_text_with_ocr(UploadFile(filename=file_for_text.filename, file=io.BytesIO(file_bytes)))
        except HTTPException as e:
            logger.warning(f"PyPDF extraction failed: {e.detail}, falling back to Azure OCR")
            try:
                text = await extract_text_with_ocr(UploadFile(filename=file_for_text.filename, file=io.BytesIO(file_bytes)))
            except Exception as ocr_error:
                logger.error(f"OCR also failed: {str(ocr_error)}", exc_info=True)
                text = ""  # If OCR also fails, return empty

        # If text extraction failed, save partial data and return
        if not text:
            logger.warning("Text extraction failed, storing file only.")
            profile_dict = {
                "name": name,
                "email": email,
                "phone": phone,
                "resume_url": resume_url,
                "resume_text": "",
                "candidate_source": candidate_source,
                "created_at": datetime.utcnow()
            }
            result = await collection.insert_one(profile_dict)
            application_id = str(result.inserted_id)

            # Update user_accounts with new application_id
            await user_accounts.update_one(
                {"email": email},
                {
                    "$set": {"email": email, "name": name, "phone": phone},
                    "$addToSet": {"application_ids": application_id}
                },
                upsert=True
            )

            return JSONResponse(
                status_code=200,
                content={
                    "status": False,
                    "message": "Could not extract text from the PDF",
                    "resume_url": resume_url,
                    "user_id": application_id
                }
            )
        # Parse resume using Azure OpenAI
        parsed_data = await parse_resume_with_azure(text)

        # Store full resume profile in DB
        profile_dict = {
            "name": name,
            "email": email,
            "phone": phone,
            "resume_url": resume_url,
            "resume_text": text,
            "candidate_source": candidate_source,
            "created_at": datetime.utcnow()
        }
        result = await collection.insert_one(profile_dict)
        application_id = str(result.inserted_id)

        # Update user_accounts with new application_id
        await user_accounts.update_one(
            {"email": email},
            {
                "$set": {"email": email, "name": name, "phone": phone},
                "$addToSet": {"application_ids": application_id}
            },
            upsert=True
        )

        logger.info("Successfully processed resume and created profile")
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "user_id": application_id,
                "data": ResumeData(**parsed_data).dict()
            }
        )

    except HTTPException as e:
        logger.error(f"HTTP error during resume parsing: {str(e)}", exc_info=True)
        return JSONResponse(status_code=e.status_code, content={"message": str(e.detail)})

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Internal server error"})

@app.post("/update-resume/")
async def update_resume(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Update resume for user if the file is readable (text can be extracted).
    Upload to blob and update resume_url and resume_text in DB.
    """
    logger.info(f"Received resume update request for user_id: {user_id}")
    try:
        content = await file.read()
        if not content:
            return JSONResponse(status_code=200, content={"message": "Uploaded file is empty or unreadable"})

        # Check user exists
        collection = db["resume_profiles"]
        user = await collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return JSONResponse(status_code=200, content={"status":False,"message": "User not found"})

        # Try extracting text before uploading
        file_for_text = UploadFile(filename=file.filename, file=io.BytesIO(content))
        resume_text = await extract_text_from_pdf(file_for_text)
        if not resume_text or not resume_text.strip():
            logger.warning("Resume text could not be extracted or is empty")
            return JSONResponse(
                status_code=200,
                content={"status":False,"message": "Resume file is not readable or has no extractable content"}
            )

        # Proceed with upload to blob only if text is valid
        file_for_blob = UploadFile(filename=file.filename, file=io.BytesIO(content))
        resume_url, resume_blob_name = await upload_to_blob_storage_resume(file_for_blob, "resume")

        # Update in Mongo
        await collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "resume_url": resume_url,
                    "resume_text": resume_text,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "message": "Resume updated successfully",
                "resume_url": resume_url
            }
        )

    except Exception as e:
        logger.error(f"Error updating resume: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

class InterviewQuestionRequest(BaseModel):
    posting_title: str
    profile_id: str

async def generate_interview_questions(profile_id: str, posting_title: str) -> List[str]:
    logger.info(f"Generating interview questions for {posting_title} position")
    
    # Fetch resume profile from database
    collection = db["resume_profiles"]
    profile = await collection.find_one({"_id": ObjectId(profile_id)})
    
    if not profile:
        logger.error(f"Profile not found for ID: {profile_id}")
        return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)
    
    # Convert ObjectId and datetime objects to strings for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    profile = convert_to_serializable(profile)
    
    # Check if we have enough parsed fields to use the parsed script
    required_fields = [
        "sales_context.sales_type",
        "sales_context.sales_motion",
        "sales_context.industries_sold_into",
        "sales_context.regions_sold_into",
        "sales_context.buyer_personas",
        "role_process_exposure.sales_stages_owned",
        "role_process_exposure.average_deal_size",
        "role_process_exposure.sales_cycle_length",
        "role_process_exposure.own_quota",
        "role_process_exposure.position_level",
        "career_overview.company_history"
    ]
    
    filled_fields = 0
    for field in required_fields:
        parts = field.split('.')
        value = profile
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
                if value and (isinstance(value, list) and len(value) > 0 or isinstance(value, str) and value.strip()):
                    filled_fields += 1
                    break
            else:
                break
    
    recent_companies = []
    if "career_overview" in profile and "company_history" in profile["career_overview"]:
        company_history = profile["career_overview"]["company_history"]
        recent_companies = company_history[-3:] if len(company_history) > 3 else company_history
    
    short_tenures = []
    if recent_companies:
        for company in recent_companies:
            if company.get("duration_months", 0) < 15:
                short_tenures.append(company)
    
    questions = []
    
    if filled_fields >= 5:
        # Use Parsed Script
        role = profile.get("role_process_exposure", {}).get("sales_role_type", "sales role")
        motion = profile.get("sales_context", {}).get("sales_motion", "sales")
        if isinstance(motion, list):
            motion = " and ".join(motion)
        deal_size = profile.get("role_process_exposure", {}).get("average_deal_size", "various deal sizes")
        buyers = profile.get("sales_context", {}).get("buyer_personas", [])
        industries = profile.get("sales_context", {}).get("industries_sold_into", [])

        buyer_str = ", ".join(buyers[:1]) if buyers else "a buyer"
        industry_str = ", ".join(industries[:1]) if industries else "an industry"

        # Q1 – Role and Day-to-Day
        questions.append(
            f"Looks like your last role was as a {role}, mostly working {motion} leads, with deal sizes around {deal_size}. "
            f"Can you walk me through what your day actually looked like, and what parts of the process you personally handled?"
        )

        # Q2 – Deal or Proud Moment
        questions.append(
            f"Can you tell me about a deal or customer interaction you’re proud of? "
            f"Maybe something with a {buyer_str} or in {industry_str} — but totally your call."
        )

        # Q3 – Interest in the Role
        questions.append(
            "What made you interested in this role or company? Was there anything specific that caught your eye or felt like a good fit?"
        )

        # Q4 – Open Space
        questions.append(
            "This one’s your space. Is there anything you’d like to share — maybe a pivot, a break, or even something exciting that doesn’t show up on your resume?"
        )

        # Q5 – Future Fit and Close
        questions.append(
            "Last one — what kind of role, team, or setup brings out your best? Anything you’re hoping for in your next move?"
        )

        # Optional short tenure question
        if len(short_tenures) >= 2:
            questions.append(
                "I noticed a couple short stints — totally up to you, but happy to hear the story if you'd like to share."
            )
    else:
        # Use Fallback Script
        questions = [
            "Can you walk me through your last sales role — what kind of company it was, and what your day-to-day looked like?",
            "Can you tell me about a deal or customer interaction you’re proud of?",
            "What made you interested in this role or company? Was there anything specific that caught your eye or felt like a good fit?",
            "This one’s your space. Is there anything you’d like to share — maybe a pivot, a break, or even something exciting that doesn’t show up on your resume?",
            "Last one — what kind of role, team, or setup brings out your best? Anything you’re hoping for in your next move?"
        ]

        if len(short_tenures) >= 2:
            questions.append(
                "I noticed a couple short stints — totally up to you, but happy to hear more if you'd like to share."
            )

    logger.info("Successfully generated interview questions")
    return questions

@app.post("/gggggggenerate-interview-questions/")
async def generate_questions(request: InterviewQuestionRequest):
    """
    Generate interview questions based on stored resume data and job requirements
    """
    logger.info(f"Received request to generate questions for {request.posting_title} position")
    try:
        collection = db["resume_profiles"]
        profile = await collection.find_one({"_id": ObjectId(request.profile_id)})
        
        if not profile:
            return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)
        if "interview_otp" in profile:
            return JSONResponse(content={"message": "Audio round already completed","status":False}, status_code=200)
        questions = await generate_interview_questions(
            request.profile_id,
            request.posting_title
        )
        return {"questions": questions}
    except Exception as e:
        error_msg = f"Error generating questions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class QAPair(BaseModel):
    question: str
    answer: str

class DimensionScore(BaseModel):
    score: int
    feedback: str

class AnswerEvaluation(BaseModel):
    credibility_score: int  # 0–100
    communication_score: int  # 0–100
    sales_motion: Literal["inbound", "outbound", "hybrid", "not mentioned"]
    sales_cycle: Literal["short", "medium", "long", "not mentioned"]
    icp: str  # e.g., "SMBs in retail, marketing managers"
    highlights: List[str]  # 2–3 clear strengths
    red_flags: List[str]  # can be empty
    coaching_focus: Optional[str]  # skill to improve (mainly from Q6)
    fit_summary: str  # 2–3 sentence narrative

class InterviewEvaluation(BaseModel):
    qa_pairs: List[QAPair]
    user_id: str

async def evaluate_answer(question: str, answer: str) -> AnswerEvaluation:
    logger.info(f"Evaluating answer for question: {question[:100]}...")

    prompt = f"""
Evaluate the following interview Q&A based on the detailed scoring framework and return a JSON response.

Question: {question}
Answer: {answer}

### Scoring Framework ###

Credibility Score (0–100)
Weighting:
- 30% Specificity: Mentions of tools, buyers, metrics, processes, volumes.
- 30% Consistency: Matches resume claims (dates, metrics, tools).
- 20% Plausibility: Claims realistic for role/tenure.
- 20% Ownership: Uses “I” and describes personal contributions.

Guidance:
- 80–100: High credibility — strong, consistent detail.
- 60–79: Mostly credible, some gaps.
- <60: Inconsistent or vague.

Communication Score (0–100)
Weighting:
- 40% Clarity & Structure: Logical, easy to follow.
- 30% Relevance: Answers the actual question.
- 20% Brevity: Ideally ~45–60 seconds per answer.
- 10% Probe Responsiveness: Improved detail when probed (Q1–Q4 only).

Guidance:
- 80–100: Clear, concise, engaging.
- 60–79: Understandable but sometimes unclear or verbose.
- <60: Hard to follow, off-topic, vague.

### Required Output (JSON Format) ###
{{
  "credibility_score": int (0–100),
  "communication_score": int (0–100),
  "sales_motion": "inbound" | "outbound" | "hybrid" | "not mentioned",
  "sales_cycle": "short" | "medium" | "long" | "not mentioned",
  "icp": "string",  # Mention company sizes, industries, buyer roles
  "highlights": ["string", "string", ...],  # 2–3 clear strengths
  "red_flags": ["string", "string", ...],   # if any
  "coaching_focus": "string",  # Skill to improve (based on Q6)
  "fit_summary": "2–3 sentence overall assessment"
}}
Only return a valid JSON object. Do not include any markdown or explanations.
"""

    try:
        async with aiohttp.ClientSession() as session:
            logger.debug("Sending request to Azure OpenAI API for answer evaluation")
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert interviewer evaluating candidate responses based on credibility, communication, and interview context. Output a JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"}
                }
            ) as response:
                if response.status != 200:
                    error_msg = f"Azure OpenAI API returned status code {response.status}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)

                result = await response.json()
                logger.debug("Received response from Azure OpenAI API")

                try:
                    content = result["choices"][0]["message"]["content"]
                    evaluation = json.loads(content)

                    required_fields = [
                        "credibility_score",
                        "communication_score",
                        "sales_motion",
                        "sales_cycle",
                        "icp",
                        "highlights",
                        "red_flags",
                        "coaching_focus",
                        "fit_summary"
                    ]

                    for field in required_fields:
                        if field not in evaluation:
                            raise ValueError(f"Missing required field: {field}")

                    # Validate numeric score ranges
                    evaluation["credibility_score"] = max(0, min(100, evaluation["credibility_score"]))
                    evaluation["communication_score"] = max(0, min(100, evaluation["communication_score"]))

                    logger.info("Successfully evaluated answer")
                    return AnswerEvaluation(**evaluation)

                except (KeyError, json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Error parsing Azure OpenAI response: {str(e)}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error evaluating answer: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

async def send_interview_result_email(email: str, passed: bool, otp: Optional[str] = None):
    """
    Send email notification about interview results
    """
    try:
        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        
        if passed:
            subject = "Congratulations! You've Passed the Audio Interview"
            content = f"""
            <p>Dear Candidate,</p>
            <p>Congratulations! You have successfully passed the audio interview round.</p>
            <p>Your interview verification code is: <strong>{otp}</strong></p>
            <p>Please keep this code safe as you'll need it for the next round.</p>
            <p>Click on the below link to take the video interview round enter your email and code to continue.</p>
            <p>https://scooter-ai-frontend.vercel.app/interview/communication?verify={otp}</p>
            <p>Best regards,<br>Team Scooter</p>
            """
        else:
            subject = "Audio Interview Results"
            content = """
            <p>Dear Candidate,</p>
            <p>Thank you for completing the audio interview. Unfortunately, we will not be moving forward with your application at this time.</p>
            <p>We appreciate your interest and wish you the best in your job search.</p>
            <p>Best regards,<br>Team Scooter</p>
            """
        
        message = Mail(
            from_email=settings.FROM_EMAIL,
            to_emails=email,
            subject=subject,
            html_content=content
        )
        
        response = sg.send(message)
        logger.info(f"Email sent successfully to {email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

@app.post("/evaluate-interview/")
async def evaluate_interview(evaluation: InterviewEvaluation):
    """
    Evaluate interview Q&A pairs and provide detailed feedback.
    Also updates the audio interview status in the database based on average score.
    """
    logger.info("Received interview evaluation request")
    try:
        collection = db["resume_profiles"]
        audio_results_collection = db["audio_interview_results"]
        profile = await collection.find_one({"_id": ObjectId(evaluation.user_id)})

        if not profile:
            return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)

        results = []
        credibility_scores = []
        communication_scores = []
        strengths = []
        areas_for_improvement = []
        red_flags = []
        sales_motions = []
        sales_cycles = []
        icp_entries = []
        coaching_focus = None

        for idx, qa_pair in enumerate(evaluation.qa_pairs):
            evaluation_result = await evaluate_answer(qa_pair.question, qa_pair.answer)
            results.append({
                "question": qa_pair.question,
                "answer": qa_pair.answer,
                "evaluation": evaluation_result.dict()
            })

            credibility_scores.append(evaluation_result.credibility_score)
            communication_scores.append(evaluation_result.communication_score)

            # Collect supporting data
            if evaluation_result.communication_score >= 80 or evaluation_result.credibility_score >= 80:
                strengths.append(evaluation_result.fit_summary)

            if evaluation_result.communication_score < 60 or evaluation_result.credibility_score < 60:
                areas_for_improvement.append(evaluation_result.fit_summary)

            if evaluation_result.red_flags:
                red_flags.extend(evaluation_result.red_flags)

            if evaluation_result.sales_motion != "not mentioned":
                sales_motions.append(evaluation_result.sales_motion)

            if evaluation_result.sales_cycle != "not mentioned":
                sales_cycles.append(evaluation_result.sales_cycle)

            if evaluation_result.icp:
                icp_entries.append(evaluation_result.icp)

            if idx == 5:  # Q6 is for coaching focus
                coaching_focus = evaluation_result.coaching_focus

        # Compute averages
        avg_credibility = round(sum(credibility_scores) / len(credibility_scores), 2) if credibility_scores else 0
        avg_communication = round(sum(communication_scores) / len(communication_scores), 2) if communication_scores else 0
        avg_total_score = round((avg_credibility + avg_communication) / 2, 2)

        audio_interview_status = avg_total_score >= 65  # New threshold on 0–100 scale

        # Generate OTP
        import string, random
        otp = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

        # Update resume profile with status
        await collection.update_one(
            {"_id": ObjectId(evaluation.user_id)},
            {"$set": {
                "audio_interview": True,
                "interview_otp": otp,
                "video_interview_start": False
            }}
        )

        # Store detailed evaluation
        evaluation_doc = {
            "user_id": evaluation.user_id,
            "qa_evaluations": results,
            "interview_summary": {
                "average_score": avg_total_score,
                "credibility_score": avg_credibility,
                "communication_score": avg_communication,
                "total_questions": len(results),
                "strengths": strengths,
                "areas_for_improvement": areas_for_improvement,
                "red_flags": red_flags,
                "icp_summary": list(set(icp_entries)),
                "sales_motion_summary": list(set(sales_motions)),
                "sales_cycle_summary": list(set(sales_cycles)),
                "coaching_focus": coaching_focus,
                "audio_interview_status": audio_interview_status
            },
            "created_at": datetime.utcnow()
        }

        await audio_results_collection.insert_one(evaluation_doc)

        return {
            "status": True,
            "message": "Thank you for completing the audio interview. Your answers have been recorded.",
            "qualified_for_video_round": audio_interview_status
        }

    except Exception as e:
        error_msg = f"Error evaluating interview: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class StructuredInterviewRequest(BaseModel):
    resume_content: str
    role: str

@app.post("/structured-ai-interview-questions/")
async def structured_ai_interview_questions(file: UploadFile = File(...), role: str = Form(...)):
    """
    Generate structured AI interview questions for Grit, Learning Agility, and Commercial Acumen traits.
    Accepts a PDF file upload and a role string.
    """
    logger.info("Received request for structured AI interview questions (file upload)")
    import random

    # Extract resume text from PDF
    text = await extract_text_from_pdf(file)

    # 1. Select one guiding question for each trait
    grit_guiding = random.choice(GRIT_GUIDING_QUESTIONS)
    learning_guiding = random.choice(LEARNING_AGILITY_GUIDING_QUESTIONS)

    # 2. Generate 2-3 probing questions for each trait, referencing the resume using OpenAI
    async def probing_for_trait(trait: str, resume: str, guiding_question: str, n: int = 3):
        prompt = f"""
        Given the following resume and guiding question, generate {n} probing follow-up questions that reference specific details from the resume. The probing questions should be used if the candidate's initial answer lacks specificity, process clarity, or outcome.

        Resume:
        {resume}

        Guiding Question:
        {guiding_question}

        Important: All questions must be sales-focused, even if the candidate's background is in other fields. Frame questions around:
        1. Sales achievements and metrics
        2. Deal closing and negotiation
        3. Client relationship building
        4. Sales process and methodology
        5. Revenue generation and business impact
        6. Market analysis and competitive positioning
        7. Sales pipeline management
        8. Quota attainment and territory management

        Return ONLY a JSON array of probing questions.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    AZURE_OPENAI_URL,
                    headers=AZURE_HEADERS,
                    json={
                        "messages": [
                            {"role": "system", "content": "You are a thoughtful sales hiring manager."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.5,
                        "max_tokens": 300,
                        "response_format": {"type": "json_object"}
                    }
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to generate probing questions for {trait}")
                        return [f"Could not generate probing questions for {trait}."]
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        questions = json.loads(content)
                        if isinstance(questions, dict):
                            questions = list(questions.values())[0]
                        return questions
                    except Exception as e:
                        logger.error(f"Error parsing probing questions: {str(e)}")
                        return [f"Could not generate probing questions for {trait}."]
        except Exception as e:
            logger.error(f"Error generating probing questions: {str(e)}")
            return [f"Could not generate probing questions for {trait}."]

    grit_probing = await probing_for_trait("grit", text, grit_guiding, n=3)
    learning_probing = await probing_for_trait("learning agility", text, learning_guiding, n=3)

    # 3. Generate Commercial Acumen scenario-based question using OpenAI (Azure)
    scenario_prompt = f"""
    You are a thoughtful hiring manager preparing a situational interview question for a sales role focused on Commercial Acumen. The candidate's resume is below. Write ONE scenario-based, job-specific question that assesses commercial acumen for the role of {role}. Do not reference the resume directly in the main question. Keep the tone warm, professional, and curious.

    Resume:
    {text}
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a thoughtful hiring manager."},
                        {"role": "user", "content": scenario_prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"}
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Failed to generate Commercial Acumen question")
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    commercial_question = json.loads(content)
                    if isinstance(commercial_question, dict):
                        commercial_question = list(commercial_question.values())[0]
                except Exception:
                    commercial_question = content.strip()
    except Exception as e:
        logger.error(f"Error generating Commercial Acumen question: {str(e)}")
        commercial_question = "Imagine you are responsible for pricing a new product in a competitive market. How would you approach the decision to maximize both revenue and customer satisfaction?"

    commercial_probing = await probing_for_trait("commercial acumen", text, str(commercial_question), n=3)

    closing_prompt = (
        "Thanks for sharing your experiences. We appreciate the time and thought you've put into these responses.\n"
        "If there's anything else you'd like us to know about how you approach challenges, learning, or business impact — feel free to add it here."
    )

    return {
        "Trait 1: Grit": {
            "Guiding Question (Trait-Focused)": grit_guiding,
            "Probing Questions (CV-Aware, to use if needed)": grit_probing
        },
        "Trait 2: Learning Agility": {
            "Guiding Question (Trait-Focused)": learning_guiding,
            "Probing Questions (CV-Aware, to use if needed)": learning_probing
        },
        "Trait 3: Commercial Acumen": {
            "Question (situational – job-specific scenario)": commercial_question,
            "Probing Questions (CV-Aware, to use if needed)": commercial_probing
        },
        "Closing Prompt": closing_prompt
    }

def validate_resume_file(file: UploadFile) -> None:
    """
    Validate if the uploaded file is a resume.
    Checks both content type and file extension.
    """
    # Check content type
    if not file.content_type == 'application/pdf':
        return JSONResponse(status_code=200, content={"message": "Only PDF files are allowed"})
        
    
    # Check file extension
    allowed_extensions = {'.pdf'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        return JSONResponse(status_code=200, content={"message": "Only PDF files are allowed"})

# async def evaluate_intervieww(session_id: str, answers: list, resume_text: str, role: str):
#     """
#     Evaluate the interview based on the 6 Sales Manager questions and generate summary with recommendation.
#     Saves detailed evaluation to database exactly as before.
#     """
#     interview_questions = [
#         {
#             "question": "You visit a potential client's office, but the security guard won't let you in. You don't have a contact inside. What would you do? Feel free to share how you usually handle situations like this on the ground."
#         },
#         {
#             "question": "You're pitching a company and they say, 'Your products are more expensive than our current supplier.' How would you respond? You can also tell us what's worked for you in the past when facing price objections."
#         },
#         {
#             "question": "It's been a tough week — lots of rejections, and you haven't closed any deals. It's Friday evening. What would you do to finish strong? You can also tell us how you usually stay motivated during slow weeks."
#         },
#         {
#             "question": "A client says, 'We've been working with the same supplier for 10 years. Why should we switch to you?' How would you build trust and make them consider Sai Marketing?"
#         },
#         {
#             "question": "You meet someone who doesn't know Sai Marketing. They ask, 'What do you do?' Explain our company and what we offer in under 30 seconds — just like you would in a real sales conversation."
#         }
#         # {
#         #     "question": "Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?"
#         # }
#     ]

#     # Filter regular answers (exclude final thoughts)
#     regular_answers = [
#         answer for answer in answers 
#         if answer.get("type") != "final_thoughts" and "question_number" in answer
#     ]

#     evaluated_questions = []
#     total_skill_score = 0
#     total_trait_score = 0
#     valid_evaluations = 0

#     for answer in regular_answers:
#         q_num = answer["question_number"]
#         user_answer = answer.get("answer", "").strip()

#         # Get corresponding question text
#         try:
#             question_text = interview_questions[q_num - 1]["question"]
#         except IndexError:
#             question_text = answer.get("question", f"Question {q_num}")

#         if user_answer:
#             evaluation = await evaluate_question_openai(
#                 question_text,
#                 user_answer,
#                 q_num,
#                 resume_text,
#                 role
#             )
            
#             # Structure evaluation data exactly as before
#             question_evaluation = {
#                 "question_number": q_num,
#                 "question": question_text,
#                 "answer": user_answer,
#                 "skill_score": evaluation.get("skill_score", 0),
#                 "trait_score": evaluation.get("trait_score", 0),
#                 "skill_reasoning": evaluation.get("skill_reasoning", ""),
#                 "trait_reasoning": evaluation.get("trait_reasoning", ""),
#                 "has_signal": evaluation.get("has_signal", True),
#                 "timestamp": answer.get("timestamp", datetime.utcnow())
#             }
#             evaluated_questions.append(question_evaluation)

#             if evaluation.get("has_signal", True):
#                 total_skill_score += evaluation.get("skill_score", 0)
#                 total_trait_score += evaluation.get("trait_score", 0)
#                 valid_evaluations += 1
#         else:
#             # Handle missing answers exactly as before
#             question_evaluation = {
#                 "question_number": q_num,
#                 "question": question_text,
#                 "answer": user_answer,
#                 "skill_score": 0,
#                 "trait_score": 0,
#                 "skill_reasoning": "No answer provided",
#                 "trait_reasoning": "No answer provided",
#                 "has_signal": False,
#                 "timestamp": answer.get("timestamp", datetime.utcnow())
#             }
#             evaluated_questions.append(question_evaluation)

#     # Calculate averages exactly as before
#     avg_skill_score = total_skill_score / valid_evaluations if valid_evaluations > 0 else 0
#     avg_trait_score = total_trait_score / valid_evaluations if valid_evaluations > 0 else 0

#     # Generate summary and recommendation
#     summary = await generate_interview_summary(
#         evaluated_questions, avg_skill_score, avg_trait_score, resume_text, role
#     )

#     # Structure final evaluation result exactly as before
#     evaluation_result = {
#         "session_id": session_id,
#         "question_evaluations": evaluated_questions,
#         "overall_scores": {
#             "average_skill_score": round(avg_skill_score, 2),
#             "average_trait_score": round(avg_trait_score, 2),
#             "total_questions": len(interview_questions)+1,
#             "questions_with_signal": valid_evaluations,
#             "questions_answered": len(regular_answers)
#         },
#         "summary": summary,
#         "interview_completed": True,
#         "evaluation_timestamp": datetime.utcnow(),
#         "role": role
#     }

#     return evaluation_result

# async def evaluate_question_openai(question: str, answer: str, question_number: int, resume_text: str, role: str):
#     """
#     Use Azure OpenAI to evaluate a single question-answer pair for skill and trait scores.
#     Updated for Sales Manager role at Sai Marketing.
#     """
#     prompt = f"""
#     You are evaluating a sales interview candidate for the Sales Manager role at Sai Marketing in Chennai.
#     This is a field-heavy, client-facing role focused on building Sai's presence in Chennai.
    
#     Question {question_number}: {question}
    
#     Candidate's Answer: {answer}
    
#     Resume Context: {resume_text[:1000]}...
    
#     Please evaluate this answer on two dimensions:
    
#     1. SKILL (1-5): Practical sales skills including:
#        - Field sales effectiveness and ground-level tactics
#        - Client relationship building and trust development
#        - Objection handling and price negotiation
#        - Territory management and prospecting
#        - Clear communication and presentation skills
#        - Problem-solving in real sales situations
    
#     2. TRAIT (1-5): Personal traits including:
#        - Grit (persistence, resilience in face of rejection)
#        - Initiative (proactive approach, taking ownership)
#        - Adaptability (learning from feedback, adjusting approach)
#        - Drive and motivation (staying motivated during tough times)
#        - Resourcefulness (finding creative solutions)
#        - Professional attitude and confidence
    
#     If the answer is too vague, unclear, off-topic, or doesn't demonstrate the required skills/traits, mark as "No Signal".
    
#     Consider the field sales context - this role involves:
#     - Direct client visits and face-to-face meetings
#     - Building relationships with new prospects
#     - Competing against established suppliers
#     - Representing Sai Marketing's brand and values
    
#     Return your evaluation in this JSON format:
#     {{
#         "skill_score": [1-5 or 0 for no signal],
#         "trait_score": [1-5 or 0 for no signal], 
#         "skill_reasoning": "Brief explanation of skill score based on sales competencies",
#         "trait_reasoning": "Brief explanation of trait score based on personal characteristics",
#         "has_signal": [true/false]
#     }}
    
#     Focus on practical sales ability, real-world experience, and the mindset needed for field sales success.
#     """
    
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 AZURE_OPENAI_URL,
#                 headers=AZURE_HEADERS,
#                 json={
#                     "messages": [
#                         {"role": "system", "content": "You are an expert sales interviewer evaluating candidates for field sales roles."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     "temperature": 0.3,
#                     "max_tokens": 500,
#                     "response_format": {"type": "json_object"}
#                 }
#             ) as response:
#                 if response.status != 200:
#                     print(f"Azure OpenAI API error: {response.status}")
#                     raise Exception("Failed to evaluate question")
                
#                 result = await response.json()
#                 content = result["choices"][0]["message"]["content"]
                
#                 try:
#                     evaluation = json.loads(content)
                    
#                     # Validate and sanitize scores
#                     skill_score = evaluation.get("skill_score", 0)
#                     trait_score = evaluation.get("trait_score", 0)
#                     has_signal = evaluation.get("has_signal", True)
                    
#                     # Ensure scores are in valid range
#                     if has_signal:
#                         skill_score = max(1, min(5, skill_score))
#                         trait_score = max(1, min(5, trait_score))
#                     else:
#                         skill_score = 0
#                         trait_score = 0
                    
#                     return {
#                         "skill_score": skill_score,
#                         "trait_score": trait_score,
#                         "skill_reasoning": evaluation.get("skill_reasoning", ""),
#                         "trait_reasoning": evaluation.get("trait_reasoning", ""),
#                         "has_signal": has_signal
#                     }
                    
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing evaluation response: {e}")
#                     raise Exception("Failed to parse evaluation response")
                
#     except Exception as e:
#         print(f"Error evaluating question: {e}")
#         return {
#             "skill_score": 0,
#             "trait_score": 0,
#             "skill_reasoning": "Evaluation error",
#             "trait_reasoning": "Evaluation error", 
#             "has_signal": False
#         }
# async def generate_interview_summary(evaluated_questions: list, avg_skill_score: float, avg_trait_score: float, resume_text: str, role: str):
#     """
#     Generate overall summary and recommendation based on all evaluations using Azure OpenAI.
#     """
#     # Prepare evaluation data for summary
#     evaluation_data = []
#     for eq in evaluated_questions:
#         evaluation_data.append(f"Q{eq['question_number']}: Skill={eq['skill_score']}, Trait={eq['trait_score']}, Signal={eq['has_signal']}")
    
#     evaluation_summary = "\n".join(evaluation_data)
    
#     prompt = f"""
#     You are summarizing a sales interview evaluation for a {role} candidate.
    
#     Individual Question Evaluations:
#     {evaluation_summary}
    
#     Overall Scores:
#     - Average Skill Score: {avg_skill_score}/5
#     - Average Trait Score: {avg_trait_score}/5
    
#     Resume Context: {resume_text[:800]}...
    
#     Write a concise summary covering:
    
#     1. KEY STRENGTHS: What this candidate does well (2-3 bullet points)
#     2. CONCERNS: Any areas of concern or weakness (1-2 bullet points)  
#     3. RECOMMENDATION: Choose one:
#        - "Proceed" (Strong candidate, move forward)
#        - "Maybe" (Mixed signals, needs further evaluation)
#        - "Do not proceed" (Significant concerns, not a good fit)
    
#     Focus on:
#     - Clear thinking and practical action demonstrated
#     - Sales mindset and approach
#     - Coachability and growth potential
#     - Overall fit for the role
    
#     Keep the summary under 200 words and be specific about what you observed.
    
#     Format as:
#     **Key Strengths:**
#     - [strength 1]
#     - [strength 2]
    
#     **Concerns:**
#     - [concern 1]
    
#     **Recommendation:** [Proceed/Maybe/Do not proceed]
    
#     **Reasoning:** [1-2 sentences explaining the recommendation]
#     """
    
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 AZURE_OPENAI_URL,
#                 headers=AZURE_HEADERS,
#                 json={
#                     "messages": [
#                         {"role": "system", "content": "You are an expert interviewer creating candidate evaluation summaries."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     "temperature": 0.3,
#                     "max_tokens": 800
#                 }
#             ) as response:
#                 if response.status != 200:
#                     print(f"Azure OpenAI API error: {response.status}")
#                     raise Exception("Failed to generate summary")
                
#                 result = await response.json()
#                 content = result["choices"][0]["message"]["content"]
                
#                 return content.strip()
                
#     except Exception as e:
#         print(f"Error generating summary: {e}")
#         return f"""
#         **Key Strengths:**
#         - Evaluation completed with {len([eq for eq in evaluated_questions if eq['has_signal']])} valid responses
        
#         **Concerns:**
#         - Unable to generate detailed summary due to processing error
        
#         **Recommendation:** Maybe
        
#         **Reasoning:** Manual review recommended due to evaluation error.
#         """


# @app.post("/conversational-interview/")
# async def conversational_interview(
#     file: UploadFile = File(None),
#     flag: str = Form(None),
#     role: str = Form(None),
#     user_id: str = Form(None),
#     session_id: str = Form(None),
#     user_answer: str = Form(None)
# ):
#     """
#     Stateful, stepwise conversational interview endpoint.
#     - First call: file+role+user_id, returns session_id and first question.
#     - Subsequent: session_id+user_answer, returns next question or closing prompt.
#     - Final: session_id+user_answer after closing prompt, saves final thoughts.
#     """
#     collection = db["interview_sessions"]
#     resume_collection = db["resume_profiles"]
    
#     # Define the specific questions to ask in order (Sales Manager - Sai Marketing)
#     interview_questions = [
#         {
#             "question": "You visit a potential client's office, but the security guard won't let you in. You don't have a contact inside. What would you do? Feel free to share how you usually handle situations like this on the ground."
#         },
#         {
#             "question": "You're pitching a company and they say, 'Your products are more expensive than our current supplier.' How would you respond? You can also tell us what's worked for you in the past when facing price objections."
#         },
#         {
#             "question": "It's been a tough week — lots of rejections, and you haven't closed any deals. It's Friday evening. What would you do to finish strong? You can also tell us how you usually stay motivated during slow weeks."
#         },
#         {
#             "question": "A client says, 'We've been working with the same supplier for 10 years. Why should we switch to you?' How would you build trust and make them consider Sai Marketing?"
#         },
#         {
#             "question": "You meet someone who doesn't know Sai Marketing. They ask, 'What do you do?' Explain our company and what we offer in under 30 seconds — just like you would in a real sales conversation."
#         }
#         # {
#         #     "question": "Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?"
#         # }
#     ]

#     # --- First call: file+role+user_id ---
#     if flag is not None and role is not None and user_id is not None and session_id is None:
#         # Check if user exists and has passed audio interview
#         user_profile = await resume_collection.find_one({"_id": ObjectId(user_id)})
#         if not user_profile:
#             raise HTTPException(status_code=404, detail="User profile not found")
                
#         # Extract text from PDF for interview
#         resume_text = user_profile.get("resume_text") 
#         if not resume_text:
#             raise HTTPException(status_code=400, detail="Could not extract text from the PDF")
        
#         session = {
#             "user_id": user_id,
#             "role": role,
#             "resume_text": resume_text,
#             "job_id": user_profile.get("job_id", None),
#             "current_question": 0,  # Start with first question
#             "answers": [],
#             "created_at": datetime.utcnow()
#         }
#         result = await collection.insert_one(session)
#         session_id = str(result.inserted_id)
        
#         # Get first question
#         first_question = interview_questions[0]
#         await collection.update_one(
#             {"_id": ObjectId(session_id)}, 
#             {"$set": {"last_question": first_question["question"]}}
#         )
#         await resume_collection.update_one(
#             {"_id": ObjectId(user_id)},
#             {"$set": {
#                 "video_interview_start": True,
#                 #"resume_blob_name": resume_blob_name
#             }}
#         )
        
#         return {
#             "session_id": session_id, 
#             "question": first_question["question"], 
#             "step": "question"
#         }

#     # --- Subsequent calls: session_id+user_answer ---
#     if session_id is not None:
#         session = await collection.find_one({"_id": ObjectId(session_id)})
#         if not session:
#             raise HTTPException(status_code=404, detail="Session not found")
        
#         # Verify user_id matches session
#         if user_id and session.get("user_id") != user_id:
#             raise HTTPException(status_code=403, detail="Unauthorized access to session")
        
#         current_question_idx = session["current_question"]
#         resume_text = session["resume_text"]
#         role = session["role"]
#         answers = session["answers"]
#         last_question = session.get("last_question")

#         # Handle final thoughts after closing prompt
#         if current_question_idx >= len(interview_questions) and user_answer is not None:
#             # Save final thoughts
#             final_thoughts = {
#                 "question_number": 6,
#                 "question":"Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?",
#                 "answer": user_answer,
#                 "timestamp": datetime.utcnow()
#             }
#             answers.append(final_thoughts)
#             # await collection.update_one(
#             #     {"_id": ObjectId(session_id)},
#             #     {"$push": {"answers": final_thoughts}}
#             # )
#             await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"answers": answers}})
#             # Evaluate the interview
#             evaluation_result = await evaluate_intervieww(session_id, session["answers"], resume_text, role)
            
#             # Save evaluation results
#             await collection.update_one(
#                 {"_id": ObjectId(session_id)},
#                 {"$set": {
#                     "evaluation": evaluation_result,
#                     "interview_completed": True,
#                     "completed_at": datetime.utcnow()
#                 }}
#             )
            
#             return {
#                 "session_id": session_id,
#                 "step": "completed",
#                 "message": "Thanks for your time. Your responses will be reviewed by the Sai Marketing team, and they'll get back to you soon."
#             }

#         # Save answer for current question
#         if user_answer:
#             answer_entry = {
#                 "question_number": current_question_idx + 1,
#                 "question": last_question,
#                 "answer": user_answer,
#                 "timestamp": datetime.utcnow()
#             }
#             answers.append(answer_entry)
#             await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"answers": answers}})

#         # Move to next question or finish
#         next_question_idx = current_question_idx + 1
        
#         if next_question_idx < len(interview_questions):
#             # Get next question
#             next_question = interview_questions[next_question_idx]
#             await collection.update_one(
#                 {"_id": ObjectId(session_id)}, 
#                 {"$set": {
#                     "current_question": next_question_idx, 
#                     "last_question": next_question["question"]
#                 }}
#             )
#             return {
#                 "session_id": session_id, 
#                 "question": next_question["question"], 
#                 "step": "question"
#             }
#         else:
#             # All questions done - send closing prompt
#             closing_prompt = (
#                 "Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?"
#             )
#             await collection.update_one(
#                 {"_id": ObjectId(session_id)}, 
#                 {"$set": {"current_question": len(interview_questions)}}
#             )
#             return {
#                 "session_id": session_id, 
#                 "question": closing_prompt, 
#                 "step": "done"
#             }
    
#     raise HTTPException(status_code=400, detail="Invalid request. Must provide either file+role+user_id or session_id+user_answer.")
from job_configs import get_job_config_by_job_id
async def evaluate_intervieww(session_id: str, answers: list, resume_text: str, role: str, job_config: dict):
    """
    Evaluate the interview based on the job-specific questions and generate summary with recommendation.
    Saves detailed evaluation to database exactly as before.
    """
    interview_questions = job_config["interview_questions"]
    evaluation_rubric = job_config["evaluation_rubric"]
    trait_rubric = job_config["trait_rubric"]

    # Filter regular answers (exclude final thoughts)
    regular_answers = [
        answer for answer in answers
        if answer.get("type") != "final_thoughts" and "question_number" in answer
    ]

    evaluated_questions = []
    total_skill_score = 0
    total_trait_score = 0
    valid_evaluations = 0

    for answer in regular_answers:
        q_num = answer["question_number"]
        user_answer = answer.get("answer", "").strip()

        # Get corresponding question text and evaluation_type
        question_data = next((q for q in interview_questions if q["question_number"] == q_num), None)
        if question_data:
            question_text = question_data["question"]
            evaluation_type = question_data["evaluation_type"]
        else:
            question_text = answer.get("question", f"Question {q_num}")
            evaluation_type = "q_final"

        if user_answer and evaluation_type:
            evaluation = await evaluate_question_openai(
                question_text,
                user_answer,
                q_num,
                resume_text,
                role,
                evaluation_rubric.get(evaluation_type)
            )

            question_evaluation = {
                "question_number": q_num,
                "question": question_text,
                "answer": user_answer,
                "skill_score": evaluation.get("skill_score", 0),
                "trait_score": evaluation.get("trait_score", 0),
                "skill_reasoning": evaluation.get("skill_reasoning", ""),
                "trait_reasoning": evaluation.get("trait_reasoning", ""),
                "has_signal": evaluation.get("has_signal", True),
                "timestamp": answer.get("timestamp", datetime.utcnow())
            }
            evaluated_questions.append(question_evaluation)

            if evaluation.get("has_signal", True):
                total_skill_score += evaluation.get("skill_score", 0)
                total_trait_score += evaluation.get("trait_score", 0)
                valid_evaluations += 1
        else:
            question_evaluation = {
                "question_number": q_num,
                "question": question_text,
                "answer": user_answer,
                "skill_score": 0,
                "trait_score": 0,
                "skill_reasoning": "No answer provided" if not user_answer else "No specific evaluation rubric found for this question type.",
                "trait_reasoning": "No answer provided" if not user_answer else "No specific evaluation rubric found for this question type.",
                "has_signal": False,
                "timestamp": answer.get("timestamp", datetime.utcnow())
            }
            evaluated_questions.append(question_evaluation)

    avg_skill_score = total_skill_score / valid_evaluations if valid_evaluations > 0 else 0
    avg_trait_score = total_trait_score / valid_evaluations if valid_evaluations > 0 else 0

    summary = await generate_interview_summary(
        evaluated_questions, avg_skill_score, avg_trait_score, resume_text, role, job_config
    )

    evaluation_result = {
        "session_id": session_id,
        "question_evaluations": evaluated_questions,
        "overall_scores": {
            "average_skill_score": round(avg_skill_score, 2),
            "average_trait_score": round(avg_trait_score, 2),
            "total_questions": len(interview_questions)+1,
            "questions_with_signal": valid_evaluations,
            "questions_answered": len(regular_answers)
        },
        "summary": summary,
        "interview_completed": True,
        "evaluation_timestamp": datetime.utcnow(),
        "role": role
    }

    return evaluation_result

async def evaluate_question_openai(question: str, answer: str, question_number: int, resume_text: str, role: str, question_rubric: dict):
    """
    Use Azure OpenAI to evaluate a single question-answer pair for skill and trait scores,
    using the provided job-specific rubric for guidance.
    """
    rubric_str = json.dumps(question_rubric, indent=2)

    prompt = f"""
    You are evaluating a sales interview candidate for the {role} role.
    This evaluation should closely follow the specific rubric provided for this question.

    Question {question_number}: {question}

    Candidate's Answer: {answer}

    Resume Context: {resume_text[:1000]}...

    Evaluation Rubric for this question:
    {rubric_str}

    Based on the above rubric and the candidate's answer, please evaluate this answer on two dimensions:

    1. SKILL (1-5): Practical skills relevant to the question and role, strictly following 'scoring_logic' in the rubric.
    2. TRAIT (1-5): Personal traits relevant to the question and role, as indicated by 'trait_rubric' (if available, otherwise general sales traits).

    If the answer is too vague, unclear, off-topic, or doesn't demonstrate the required skills/traits, mark as "No Signal".

    Return your evaluation in this JSON format:
    {{
        "skill_score": [1-5 or 0 for no signal],
        "trait_score": [1-5 or 0 for no signal],
        "skill_reasoning": "Brief explanation of skill score based on the rubric's competencies and the answer.",
        "trait_reasoning": "Brief explanation of trait score based on the rubric's characteristics and the answer.",
        "has_signal": [true/false]
    }}

    Focus on practical ability, real-world experience, and the mindset needed for success in this role.
    When determining scores, prioritize adherence to the structured 'scoring_logic' within the rubric.
    """

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert sales interviewer evaluating candidates strictly according to provided rubrics. Calculate scores precisely based on the rubric's criteria."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                }
            ) as response:
                if response.status != 200:
                    print(f"Azure OpenAI API error: {response.status}")
                    raise Exception("Failed to evaluate question")

                result = await response.json()
                content = result["choices"][0]["message"]["content"]

                try:
                    evaluation = json.loads(content)

                    skill_score = evaluation.get("skill_score", 0)
                    trait_score = evaluation.get("trait_score", 0)
                    has_signal = evaluation.get("has_signal", True)

                    if has_signal:
                        skill_score = max(1, min(5, skill_score))
                        trait_score = max(1, min(5, trait_score))
                    else:
                        skill_score = 0
                        trait_score = 0

                    return {
                        "skill_score": skill_score,
                        "trait_score": trait_score,
                        "skill_reasoning": evaluation.get("skill_reasoning", ""),
                        "trait_reasoning": evaluation.get("trait_reasoning", ""),
                        "has_signal": has_signal
                    }

                except json.JSONDecodeError as e:
                    print(f"Error parsing evaluation response: {e}")
                    raise Exception("Failed to parse evaluation response")

    except Exception as e:
        print(f"Error evaluating question: {e}")
        return {
            "skill_score": 0,
            "trait_score": 0,
            "skill_reasoning": f"Evaluation error: {e}",
            "trait_reasoning": f"Evaluation error: {e}",
            "has_signal": False
        }

async def generate_interview_summary(evaluated_questions: list, avg_skill_score: float, avg_trait_score: float, resume_text: str, role: str, job_config: dict):
    """
    Generate overall summary and recommendation based on all evaluations and job-specific decision thresholds.
    """
    overall_decision_thresholds = job_config.get("overall_decision_thresholds", [])
    trait_rubric_config = job_config.get("trait_rubric", {})

    evaluation_data = []
    for eq in evaluated_questions:
        evaluation_data.append(f"Q{eq['question_number']}: Skill={eq['skill_score']}, Trait={eq['trait_score']}, Signal={eq['has_signal']}")

    evaluation_summary_text = "\n".join(evaluation_data)

    total_skill_score = sum(eq['skill_score'] for eq in evaluated_questions if eq['has_signal'])
    total_trait_score_sum = sum(eq['trait_score'] for eq in evaluated_questions if eq['has_signal'])

    total_overall_score = total_skill_score + total_trait_score_sum

    calculated_recommendation = "Maybe"
    calculated_reasoning = "Based on raw scores."

    for threshold in overall_decision_thresholds:
        if total_overall_score >= threshold["score_range"][0] and total_overall_score <= threshold["score_range"][1]:
            calculated_recommendation = threshold["decision"]
            calculated_reasoning = f"Overall score of {total_overall_score} falls into '{threshold['decision']}' range: {threshold['action']}."
            break


    prompt = f"""
    You are summarizing a sales interview evaluation for a {role} candidate.

    Individual Question Evaluations:
    {evaluation_summary_text}

    Overall Scores:
    - Average Skill Score: {avg_skill_score:.2f}/5
    - Average Trait Score: {avg_trait_score:.2f}/5 (This is an average of trait scores per question)
    - Total Skill Score (sum of all valid skill scores): {total_skill_score}
    - Total Trait Score (sum of all valid trait scores from questions): {total_trait_score_sum}
    - Combined Total Score: {total_overall_score}

    Resume Context: {resume_text[:800]}...

    Write a concise summary covering:

    1. KEY STRENGTHS: What this candidate does well (2-3 bullet points)
    2. CONCERNS: Any areas of concern or weakness (1-2 bullet points)
    3. RECOMMENDATION: Based on the combined total score of {total_overall_score}, and the following decision thresholds:
       {json.dumps(overall_decision_thresholds, indent=2)}
       Choose one: "Strong Hire", "Maybe", "Weak", "Reject".
    4. Flags: if the candidates answer feels like AI generated, flag it as "might be AI_generated" in the summary, if it is genune then ignore this.

    Focus on:
    - Clear thinking and practical action demonstrated
    - Sales mindset and approach
    - Coachability and growth potential
    - Overall fit for the role

    Keep the summary under 200 words and be specific about what you observed.

    Format as:
    **Key Strengths:**
    - [strength 1]
    - [strength 2]

    **Concerns:**
    - [concern 1]

    **Recommendation:** [Strong Hire/Maybe/Weak/Reject]

    **Reasoning:** [1-2 sentences explaining the recommendation based on overall score and specific observations from the interview.]
    """

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert interviewer creating candidate evaluation summaries based on detailed rubrics and overall scores."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }
            ) as response:
                if response.status != 200:
                    print(f"Azure OpenAI API error: {response.status}")
                    raise Exception("Failed to generate summary")

                result = await response.json()
                content = result["choices"][0]["message"]["content"]

                return content.strip()

    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"""
        **Key Strengths:**
        - Evaluation completed with {len([eq for eq in evaluated_questions if eq['has_signal']])} valid responses

        **Concerns:**
        - Unable to generate detailed summary due to processing error

        **Recommendation:** {calculated_recommendation}

        **Reasoning:** {calculated_reasoning} (Error in detailed summary generation.)
        """

@app.post("/conversational-interview/")
async def conversational_interview(
    # role: str = Form(None), # No longer strictly needed as a form param if fetched from user_profile
    user_id: str = Form(None),
    session_id: str = Form(None),
    user_answer: str = Form(None)
):
    """
    Stateful, stepwise conversational interview endpoint.
    - First call: user_id, returns session_id and first question.
    - Subsequent: session_id+user_answer, returns next question or closing prompt.
    - Final: session_id+user_answer after closing prompt, saves final thoughts.
    """
    collection = db["interview_sessions"]
    resume_collection = db["resume_profiles"]
    job_config = None

    # --- First call: user_id (role is now implicitly from user_profile.job_id) ---
    # The 'flag' and 'file' parameters from your original code were for resume upload and initial setup.
    # Assuming resume processing and user_profile creation (with job_id) happen prior to this endpoint.
    if user_id is not None and session_id is None:
        user_profile = await resume_collection.find_one({"_id": ObjectId(user_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")

        job_id = user_profile.get("job_id") # Fetch job_id from user_profile
        if not job_id:
            raise HTTPException(status_code=400, detail="Job ID not found in user profile.")
        reset_count = user_profile.get("video_interview_reset_count", 0)
        job_config = get_job_config_by_job_id(job_id) # Use the new getter
        if not job_config:
            raise HTTPException(status_code=404, detail=f"Job configuration not found for job ID: {job_id} in job_configs.py")

        interview_questions = job_config["interview_questions"]
        role_from_config = job_config["job_role"] # Get role name from config

        resume_text = user_profile.get("resume_text")
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF (resume_text missing).")

        session = {
            "user_id": user_id,
            "role": role_from_config, # Store the actual role name from config
            "job_id": job_id, # Store the job_id
            "resume_text": resume_text,
            "current_question": 0,
            "answers": [],
            "created_at": datetime.utcnow()
        }
        result = await collection.insert_one(session)
        session_id = str(result.inserted_id)

        if not interview_questions:
             raise HTTPException(status_code=500, detail="No questions configured for this job ID.")

        processed_video_url=f'https://scootervideoconsumption.blob.core.windows.net/scooter-processed-videos/{user_id}_video_{reset_count}_master.m3u8'
        first_question = interview_questions[0]
        await collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"last_question": first_question["question"]}}
        )
        await resume_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "video_interview_start": True,
                "processed_video_url": processed_video_url
                # "resume_blob_name": resume_blob_name # Keep if you manage blob names here
            }}
        )

        return {
            "session_id": session_id,
            "question": first_question["question"],
            "step": "question"
        }

    # --- Subsequent calls: session_id+user_answer ---
    if session_id is not None:
        session = await collection.find_one({"_id": ObjectId(session_id)})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        if user_id and session.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to session")

        # Retrieve job config using the stored job_id from the session
        stored_job_id = session.get("job_id")
        if not stored_job_id:
            raise HTTPException(status_code=500, detail="Job ID missing from session for config lookup.")

        job_config = get_job_config_by_job_id(stored_job_id)
        if not job_config:
            raise HTTPException(status_code=404, detail=f"Associated job configuration not found for job ID: {stored_job_id} in job_configs.py")

        interview_questions = job_config["interview_questions"]
        role = session["role"] # Keep role from session, which was set from job_config.job_role initially


        current_question_idx = session["current_question"]
        resume_text = session["resume_text"]
        answers = session["answers"]
        last_question = session.get("last_question")

        if current_question_idx >= len(interview_questions) and user_answer is not None:
            final_thoughts = {
                "question_number": len(interview_questions) + 1,
                "question":"Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?",
                "answer": user_answer,
                "timestamp": datetime.utcnow()
            }
            answers.append(final_thoughts)
            await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"answers": answers}})
            evaluate_commu = await evaluate_communication(session_id)
            evaluation_result = await evaluate_intervieww(session_id, session["answers"], resume_text, role, job_config)
            await collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {
                    "evaluation": evaluation_result,
                    "interview_completed": True,
                    "completed_at": datetime.utcnow()
                }}
            )
            # user_id_for_eval= session.get("user_id")
            # try:
            #     logger.info(f"user_id {user_id_for_eval} - session_id {session_id}")
            #     await update_candidate_summary_from_video(user_id_for_eval)
            # except Exception as update_err:
            #     logger.error(f"Error updating candidate summary from video: {str(update_err)}")
            return {
                "session_id": session_id,
                "step": "completed",
                "message": "Thanks for your time. Your responses will be reviewed by the Interview team, and they'll get back to you soon."
            }

        if user_answer:
            answer_entry = {
                "question_number": current_question_idx + 1,
                "question": last_question,
                "answer": user_answer,
                "timestamp": datetime.utcnow()
            }
            answers.append(answer_entry)
            await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"answers": answers}})

        next_question_idx = current_question_idx + 1

        if next_question_idx < len(interview_questions):
            next_question = interview_questions[next_question_idx]
            await collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {
                    "current_question": next_question_idx,
                    "last_question": next_question["question"]
                }}
            )
            return {
                "session_id": session_id,
                "question": next_question["question"],
                "step": "question"
            }
        else:
            closing_prompt = (
                "Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?"
            )
            await collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"current_question": len(interview_questions)}}
            )
            return {
                "session_id": session_id,
                "question": closing_prompt,
                "step": "done"
            }

    raise HTTPException(status_code=400, detail="Invalid request. Must provide user_id (for first call) or session_id+user_answer (for subsequent calls).")
async def generate_commercial_question(previous_answers: List[Dict], resume_text: str, role: str) -> Optional[str]:
    """
    Generate a Commercial Acumen question based on previous answers and resume.
    Returns None if generation fails, which will trigger fallback to question bank.
    """
    try:
        # Prepare context from previous answers
        context = "\n".join([
            f"Question: {answer['question']}\nAnswer: {answer['answer']}"
            for answer in previous_answers
        ])

        prompt = f"""
        Based on the candidate's previous answers and resume, generate ONE engaging Commercial Acumen question that:
        1. References specific details from their previous responses
        2. Builds upon their demonstrated sales experience and knowledge
        3. Assesses their business acumen in a sales context
        4. Maintains a warm, conversational tone

        The question MUST be sales-focused, covering one or more of these areas:
        - Deal strategy and negotiation
        - Revenue growth and quota attainment
        - Market analysis and competitive positioning
        - Sales pipeline management
        - Client relationship building
        - Territory management
        - Sales process optimization
        - Business impact and ROI

        Previous Interview Context:
        {context}

        Resume:
        {resume_text}

        Role:
        {role}

        Return ONLY the question as plain text. Make it sound natural and conversational.
        If you cannot generate a relevant question based on the context, return "FALLBACK".
        """
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert sales interviewer who asks thoughtful, context-aware questions about commercial acumen."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            ) as response:
                if response.status != 200:
                    return None
                    
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Return None if generation failed or model requested fallback
                if content == "FALLBACK" or not content:
                    return None
                    
                return content
    except Exception as e:
        logger.error(f"Error generating commercial question: {str(e)}")
        return None

# --- Helper functions for OpenAI ---
async def generate_openai_question(prompt: str) -> str:
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    AZURE_OPENAI_URL,
                    headers=AZURE_HEADERS,
                    json={
                        "messages": [
                            {"role": "system", "content": "You are an engaging and thoughtful hiring manager who asks personalized, relevant questions. Keep the tone warm and conversational."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 300,
                        "response_format": {"type": "json_object"}
                    }
                ) as response:
                    if response.status != 200:
                        retry_count += 1
                        if retry_count == max_retries:
                            return "Could not generate question. Please try again."
                        continue
                        
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        q = json.loads(content)
                        if isinstance(q, dict):
                            q = list(q.values())[0]
                        if isinstance(q, list):
                            return q[0]
                        return str(q)
                    except Exception:
                        return content.strip()
        except Exception as e:
            logger.error(f"Error generating question (attempt {retry_count + 1}): {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                return "Could not generate question. Please try again."
    
    return "Could not generate question. Please try again."

async def generate_openai_probing(trait: str, resume: str, guiding_question: str, user_answer: str) -> str:
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            prompt = f"""
            Given the following resume, guiding question, and candidate answer, generate ONE engaging probing follow-up question that:
            1. References specific details from the resume and answer
            2. Shows genuine interest in the candidate's experience
            3. Helps clarify or expand on their response
            4. Maintains a warm, conversational tone

            Resume:
            {resume}

            Guiding Question:
            {guiding_question}

            Candidate Answer:
            {user_answer}

            Return ONLY the probing question as plain text. Make it sound natural and conversational.
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    AZURE_OPENAI_URL,
                    headers=AZURE_HEADERS,
                    json={
                        "messages": [
                            {"role": "system", "content": "You are an engaging interviewer who asks thoughtful follow-up questions to better understand candidates."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 200
                    }
                ) as response:
                    if response.status != 200:
                        retry_count += 1
                        if retry_count == max_retries:
                            return "Could not generate probing question. Please try again."
                        continue
                        
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
        except Exception as e:
            logger.error(f"Error generating probing question (attempt {retry_count + 1}): {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                return "Could not generate probing question. Please try again."
    
    return "Could not generate probing question. Please try again."

async def needs_probing_openai(guiding_question: str, user_answer: str, resume: str) -> bool:
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            prompt = f"""
            Given the following guiding question, candidate answer, and resume, evaluate if the answer needs probing because it:
            1. Lacks specific details or examples
            2. Doesn't clearly explain the process or approach
            3. Doesn't mention concrete outcomes or results
            4. Could benefit from more context or clarification

            Guiding Question:
            {guiding_question}

            Candidate Answer:
            {user_answer}

            Resume:
            {resume}

            Return ONLY 'yes' if probing is needed, or 'no' if the answer is sufficiently detailed and clear.
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    AZURE_OPENAI_URL,
                    headers=AZURE_HEADERS,
                    json={
                        "messages": [
                            {"role": "system", "content": "You are an expert interviewer evaluating answer quality."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": 10
                    }
                ) as response:
                    if response.status != 200:
                        retry_count += 1
                        if retry_count == max_retries:
                            return False
                        continue
                        
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"].strip().lower()
                    return content.startswith('y')
        except Exception as e:
            logger.error(f"Error evaluating need for probing (attempt {retry_count + 1}): {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                return False
    
    return False

class CommunicationEvaluation(BaseModel):
    content_and_thought: dict
    verbal_delivery: dict
    non_verbal: dict
    presence_and_authenticity: dict
    overall_score: float
    summary: str

#@app.post("/evaluate-communication/{session_id}")
async def evaluate_communication(session_id: str):
    """
    Evaluate communication skills based on interview answers stored in the database.
    """
    logger.info(f"Evaluating communication for session {session_id}")
    
    try:
        # Fetch session from database
        collection = db["interview_sessions"]
        session = await collection.find_one({"_id": ObjectId(session_id)})
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        user_id = session.get("user_id", None)
        # Extract answers from session
        answers = session.get("answers", [])
        if not answers:
            raise HTTPException(status_code=400, detail="No answers found in session")
        
        # Prepare answers for evaluation
        answers_text = "\n".join([
            f"Question: {answer.get('question', '')}\nAnswer: {answer.get('answer', '')}"
            for answer in answers if answer.get('type') != 'final_thoughts'
        ])
        
        # Create evaluation prompt
        prompt = f"""
        Evaluate the following interview answers based on communication criteria. Return a JSON object with scores (1-5) and feedback for each category.

        Interview Answers:
        {answers_text}

        Evaluate based on these categories:

        1. Content & Thought Structure:
           - Clarity of thought (logical structure, relevance, completeness)
           - Score 1: Rambling, no sequence
           - Score 5: Clear structure with logical flow

        2. Verbal Delivery:
           - Clarity of speech (vocabulary, grammar, filler minimization)
           - Audience awareness (tailoring response to context)
           - Score 1: Filler-heavy, vague language
           - Score 5: Precise, persuasive wording

        3. Non-Verbal:
           - Tone & energy (confidence, vocal variation)
           - Pacing (natural speech tempo)
           - Facial expressions (expressiveness matching content)
           - Score 1: Flat voice, rushed or too slow
           - Score 5: Natural pace, appropriate emphasis

        4. Presence & Authenticity:
           - Natural tone, self-awareness, relatability
           - Score 1: Over-rehearsed, feels fake
           - Score 5: Warm, human, authentic

        Return a JSON object in this format:
        {{
            "content_and_thought": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "verbal_delivery": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "non_verbal": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "presence_and_authenticity": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "overall_score": float,
            "summary": "string"
        }}
        """
        
        # Call Azure OpenAI for evaluation
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert interviewer evaluating communication skills."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"}
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Failed to evaluate communication")
                
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    evaluation = json.loads(content)
                    
                    # Validate scores are within range
                    for category in ["content_and_thought", "verbal_delivery", "non_verbal", "presence_and_authenticity"]:
                        if not 1 <= evaluation[category]["score"] <= 5:
                            evaluation[category]["score"] = max(1, min(5, evaluation[category]["score"]))
                    
                    # Calculate overall score as average of category scores
                    category_scores = [
                        evaluation["content_and_thought"]["score"],
                        evaluation["verbal_delivery"]["score"],
                        evaluation["non_verbal"]["score"],
                        evaluation["presence_and_authenticity"]["score"]
                    ]
                    evaluation["overall_score"] = round(sum(category_scores) / len(category_scores), 2)
                    
                    # Save evaluation to database
                    await collection.update_one(
                        {"_id": ObjectId(session_id)},
                        {"$set": {"communication_evaluation": evaluation}}
                    )
                    
                    return JSONResponse(status_code=200, content={"status": "true","message": "evaluated successfully" ,"user_id":user_id} )
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing evaluation response: {str(e)}")
                    raise HTTPException(status_code=500, detail="Failed to parse evaluation response")
                
    except Exception as e:
        error_msg = f"Error evaluating communication: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class SalesScenarioRequest(BaseModel):
    role_type: str
    buyer_profile: str
    user_id: str

class SalesScenarioResponse(BaseModel):
    session_id: str
    scenario: str
    step: str
    scenario_type: str

class EmploymentGaps(BaseModel):
    has_gaps: bool
    duration: str

class QuotaOwnership(BaseModel):
    has_quota: bool
    amount: float
    cadence: str
    attainment_history: str

class BasicInformation(BaseModel):
    full_name: str
    current_location: str
    open_to_relocation: bool
    phone_number: str
    linkedin_url: str
    email: str
    specific_phone_number: Optional[str] = None
    notice_period: str
    current_ctc: dict = Field(..., description="Format: {'currencyType': str, 'value': float}")
    expected_ctc: dict = Field(..., description="Format: {'currencyType': str, 'value': float}")

class CompanyHistory(BaseModel):
    company_name: str
    position: str
    start_date: str
    end_date: str
    duration_months: int
    is_current: bool = False

class CareerOverview(BaseModel):
    total_years_experience: float
    years_sales_experience: float
    average_tenure_per_role: float
    employment_gaps: EmploymentGaps
    promotion_history: bool
    company_history: Optional[List[CompanyHistory]] = None

class SalesContext(BaseModel):
    sales_type: str
    sales_motion: str
    industries_sold_into: List[str]
    regions_sold_into: List[str]
    buyer_personas: List[str]

class RoleProcessExposure(BaseModel):
    sales_role_type: str
    position_level: str
    sales_stages_owned: List[str]
    average_deal_size: str
    sales_cycle_length: str
    own_quota: bool
    quota_ownership: Optional[List[str]] = None
    quota_attainment: str

class ToolsPlatforms(BaseModel):
    crm_tools: List[str]
    sales_tools: List[str]

class OptionalToolsPlatforms(BaseModel):
    crm_tools: Optional[List[str]] = None
    sales_tools: Optional[List[str]] = None

class ResumeProfile(BaseModel):
    user_id: str
    job_id: Optional[str] = None
    basic_information: BasicInformation
    career_overview: CareerOverview
    sales_context: SalesContext
    role_process_exposure: RoleProcessExposure
    tools_platforms: OptionalToolsPlatforms
    created_at: Optional[datetime] = None
    video_url: Optional[str] = None
    video_uploaded_at: Optional[datetime] = None
    audio_url: Optional[str] = None
    audio_uploaded_at: Optional[datetime] = None
    #resume_url: Optional[str] = None
    #resume_blob_name: Optional[str] = None

@app.post("/add-resume-profile/", response_model=Dict[str, str])
async def add_resume_profile(profile: ResumeProfile):
    """
    Add or update a resume profile in the database.
    If user_id is provided, updates the existing record.
    Otherwise, creates a new one.
    """
    try:
        # Convert Pydantic model to dict
        profile_dict = profile.dict()
        profile_dict["created_at"] = datetime.utcnow()

        user_id = profile_dict.pop("user_id", None)

        # Validate job_id if provided
        if profile_dict.get("job_id"):
            job_collection = db["job_roles"]
            job = await job_collection.find_one({"_id": ObjectId(profile_dict["job_id"])})
            if not job:
                return JSONResponse(status_code=200, content={"message": "Job role not found"})

        # Process company history and calculate experience
        if profile_dict.get("career_overview", {}).get("company_history"):
            company_history = profile_dict["career_overview"]["company_history"]
            company_history.sort(key=lambda x: x["start_date"],reverse=True)
            if company_history:
                total_months = sum(role["duration_months"] for role in company_history)
                profile_dict["career_overview"]["average_tenure_per_role"] = round(total_months / len(company_history) / 12, 1)
                if not profile_dict["career_overview"].get("total_years_experience"):
                    profile_dict["career_overview"]["total_years_experience"] = round(total_months / 12, 1)

        collection = db["resume_profiles"]

        if user_id:
            # Update existing record
            result = await collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": profile_dict}
            )
            if result.matched_count == 0:
                return JSONResponse(status_code=404, content={"message": "User ID not found"})
            return {"profile_id": user_id}
        else:
            # Create new record
            result = await collection.insert_one(profile_dict)
            return {"profile_id": str(result.inserted_id)}

    except Exception as e:
        error_msg = f"Error creating or updating resume profile: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


async def generate_resume_summary(resume_text: str) -> str:
    prompt = f"""You are an expert recruiter.generate A narrative summary covering: total sales experience, recent role progression, key achievements with numbers, deal types/markets worked, and any red flags (tenure patterns, career gaps, etc.). if a candidate has more then one role in same company or a promotion then highlight it. donot add formatting or bullet points, just a concise paragraph.


Resume:
{resume_text}
"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": [
                    {"role": "system", "content": "You are a professional recruiter generating summaries from resumes."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 2000,
                "response_format": {"type": "text"}  
            }
        ) as response:
            if response.status != 200:
                logger.error(f"Azure OpenAI API returned status {response.status}")
                raise HTTPException(status_code=500, detail="Error calling LLM")

            result = await response.json()
            try:
                summary = result["choices"][0]["message"]["content"].strip()
                return summary
            except (KeyError, IndexError):
                raise HTTPException(status_code=500, detail="Invalid response format from LLM")
            
async def generate_job_fit_summary(resume_text: str, job_description:str) -> str:
    prompt = f"""
You are an expert recruiter. 
provide a Job Fit Assessment for this resume and Job description based on the following criteria:
Resume:
{resume_text}
Job Description:
{job_description}

Job Fit Assessment:
- Rating: HIGH / MEDIUM / LOW
- Rationale: 2–3 bullet points comparing the candidate’s background directly to the job description, focusing on:
  - Experience level
  - Industry alignment
  - Sales skills (including sales type: B2B/B2C, Inbound/Outbound)
  - Relevant markets/regions
  - Location fit (if applicable)
  - if candidate had more then one role in same company or a promotion then highlight it.

Key Data Points (use if available):
- Sales Experience: [X years]
- Quota Performance: [specific %s or achievements]
- Markets/Regions: [list]
- Sales Type: [B2B/B2C, Inbound/Outbound, etc.]
- Average Tenure: [X months/years]

Assessment Criteria:
- HIGH: Meets 80%+ of requirements with strong sales track record
- MEDIUM: Meets 60–80% with some gaps but transferable skills
- LOW: Meets <60% or has significant experience/skill gaps

Focus on: Quantified achievements, tenure stability, relevant industry/market experience, and skill alignment with the specific role requirements.

Output Format:
Job Fit Assessment: [HIGH/MEDIUM/LOW]
Rationale:
- Bullet 1
- Bullet 2
- Bullet 3
- bullet 4
"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": [
                    {"role": "system", "content": "You are a professional HR assessing candidates"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 1000,
                "response_format": {"type": "text"}  
            }
        ) as response:
            if response.status != 200:
                logger.error(f"Azure OpenAI API returned status {response.status}")
                raise HTTPException(status_code=500, detail="Error calling LLM")

            result = await response.json()
            try:
                summary = result["choices"][0]["message"]["content"].strip()
                return summary
            except (KeyError, IndexError):
                raise HTTPException(status_code=500, detail="Invalid response format from LLM")
            
class CandidateSummaryRequest(BaseModel):
    user_id: str
@app.post("/generate-resume-summary")
async def get_resume_summary(request: CandidateSummaryRequest):
    try:
        collection = db["resume_profiles"]
        job_collection = db["job_roles"]
        user = await collection.find_one({"_id": ObjectId(request.user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        resume_text = user.get("resume_text", "").strip()
        if not resume_text:
            raise HTTPException(status_code=400, detail="Resume text missing")
        job_id = user.get("job_id")
        job=await job_collection.find_one({"_id": ObjectId(job_id)}) if job_id else None
        job_description = job.get("job_description", "") if job else ""
        summary = await generate_resume_summary(resume_text)
        job_fit_assessment = await generate_job_fit_summary(resume_text, job_description)
        result = await collection.update_one(
            {"_id": ObjectId(request.user_id)},
            {"$set": {"job_fit_assessment": job_fit_assessment}}
        )
        return {
            "status": True,
            "candidate_summary": summary
        }
    except Exception as e:
        logger.error(f"Error generating candidate summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    
async def generate_candidate_summary(candidate_data: dict) -> str:
    prompt = f"""
You are an expert recruiter. Given the following candidate details, generate a concise 3–5 sentence professional summary highlighting their experience, sales exposure, industries, achievements, and tools/platforms used. Focus on clarity, professionalism, and relevance to sales/BD roles.

Candidate Details:
Full Name: {candidate_data.get("basic_information", {}).get("full_name", "")}
Location: {candidate_data.get("basic_information", {}).get("current_location", "")}
Notice Period: {candidate_data.get("basic_information", {}).get("notice_period", "")}
Total Experience: {candidate_data.get("career_overview", {}).get("total_years_experience", "")} years
Sales Experience: {candidate_data.get("career_overview", {}).get("years_sales_experience", "")} years
Roles:
{candidate_data.get("career_overview", {}).get("company_history", "")}
Sales Type: {candidate_data.get("sales_context", {}).get("sales_type", "")}
Sales Motion: {candidate_data.get("sales_context", {}).get("sales_motion", "")}
Industries: {", ".join(candidate_data.get("sales_context", {}).get("industries_sold_into", []))}
Regions: {", ".join(candidate_data.get("sales_context", {}).get("regions_sold_into", []))}
Sales Stages Owned: {", ".join(candidate_data.get("role_process_exposure", {}).get("sales_stages_owned", []))}
Deal Size: {candidate_data.get("role_process_exposure", {}).get("average_deal_size", "")}
Cycle Length: {candidate_data.get("role_process_exposure", {}).get("sales_cycle_length", "")}
CRM Tools: {", ".join(candidate_data.get("tools_platforms", {}).get("crm_tools", []))}
Sales Tools: {", ".join(candidate_data.get("tools_platforms", {}).get("sales_tools", []))}

Generate the summary without additional explanation or formatting.
"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": [
                    {"role": "system", "content": "You are a professional recruiter generating summaries from structured candidate profiles."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 200
            }
        ) as response:
            if response.status != 200:
                logger.error(f"Azure OpenAI API returned status {response.status}")
                raise HTTPException(status_code=500, detail="Error calling LLM")

            result = await response.json()
            try:
                summary = result["choices"][0]["message"]["content"].strip()
                return summary
            except (KeyError, IndexError):
                raise HTTPException(status_code=500, detail="Invalid response format from LLM")

@app.post("/generate-candidate-summary")
async def get_candidate_summary(request: CandidateSummaryRequest):
    try:
        collection = db["resume_profiles"]
        user = await collection.find_one({"_id": ObjectId(request.user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Generate summary from structured profile
        summary = await generate_candidate_summary(user)
        return {
            "status": True,
            "candidate_summary": summary
        }

    except Exception as e:
        logger.error(f"Error generating candidate summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

class CandidateSummarySaveRequest(BaseModel):
    user_id: str
    summary_content: str

@app.post("/save-candidate-summary")
async def save_candidate_summary(request: CandidateSummarySaveRequest):
    try:
        collection = db["resume_profiles"]
        
        result = await collection.update_one(
            {"_id": ObjectId(request.user_id)},
            {"$set": {"short_summary": request.summary_content}}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {
         "status": True,
         "message": "Candidate summary saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving candidate summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
@app.post("/start-sales-scenario/", response_model=SalesScenarioResponse)
async def start_sales_scenario(
    role_type: str = Form(...),
    buyer_profile: str = Form(...),
    user_id: str = Form(...)
):
    """
    Start a new sales scenario conversation.
    Returns the first scenario and a session ID for subsequent interactions.
    """
    collection = db["sales_scenarios"]
    resume_collection = db["resume_profiles"]
    interview_collection = db["interview_sessions"]
    
    # Check if user exists and has passed audio interview
    user_profile = await resume_collection.find_one({"_id": ObjectId(user_id)})
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    if not user_profile.get("audio_interview", False):
        raise HTTPException(
            status_code=403,
            detail="You need to pass the audio interview round before proceeding with the sales scenario."
        )
    
    # Get the resume text from interview_sessions collection
    interview_session = await interview_collection.find_one(
        {"user_id": user_id},
        sort=[("created_at", -1)]  # Get the most recent session
    )
    
    if not interview_session:
        raise HTTPException(status_code=404, detail="No interview session found for user")
    
    resume_text = interview_session.get("resume_text")
    if not resume_text:
        raise HTTPException(status_code=404, detail="No resume text found in interview session")
    
    session = {
        "user_id": user_id,
        "role_type": role_type,
        "buyer_profile": buyer_profile,
        "resume_text": resume_text,
        "current_scenario": 0,
        "step": "scenario",
        "responses": [],
        "created_at": datetime.utcnow()
    }
    result = await collection.insert_one(session)
    session_id = str(result.inserted_id)
    
    # Generate first scenario
    scenario, scenario_type = await generate_sales_scenario(role_type, buyer_profile)
    await collection.update_one(
        {"_id": ObjectId(session_id)}, 
        {"$set": {"last_scenario": scenario, "scenario_type": scenario_type}}
    )
    
    return {
        "session_id": session_id,
        "scenario": scenario,
        "step": "scenario",
        "scenario_type": scenario_type
    }

@app.post("/continue-sales-scenario/", response_model=SalesScenarioResponse)
async def continue_sales_scenario(
    session_id: str = Form(...),
    user_response: str = Form(...)
):
    """
    Continue an existing sales scenario conversation.
    Handles follow-up questions and progression through scenarios.
    """
    collection = db["sales_scenarios"]
    session = await collection.find_one({"_id": ObjectId(session_id)})
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    scenario_idx = session["current_scenario"]
    step = session["step"]
    resume_text = session["resume_text"]
    role_type = session["role_type"]
    buyer_profile = session["buyer_profile"]
    responses = session["responses"]
    last_scenario = session.get("last_scenario")
    scenario_type = session.get("scenario_type", "unknown")

    # Handle final thoughts after closing prompt
    if step == "done" and user_response is not None:
        final_thoughts = {
            "type": "final_thoughts",
            "content": user_response,
            "timestamp": datetime.utcnow()
        }
        await collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$push": {"responses": final_thoughts}}
        )
        return {
            "session_id": session_id,
            "scenario": "Thank you for completing the sales scenario simulation. Your final thoughts have been recorded.",
            "step": "completed",
            "scenario_type": scenario_type
        }

    # Save response for regular scenarios
    responses.append({
        "scenario": last_scenario,
        "response": user_response,
        "timestamp": datetime.utcnow()
    })
    await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"responses": responses}})

    # Decide next step
    if step == "scenario":
        # Analyze response for follow-up questions
        needs_followup = await needs_followup_openai(last_scenario, user_response)
        if needs_followup:
            # Generate follow-up question
            followup_q = await generate_sales_followup(role_type, buyer_profile, last_scenario, user_response)
            await collection.update_one(
                {"_id": ObjectId(session_id)}, 
                {"$set": {"step": "followup", "last_followup": followup_q}}
            )
            return {
                "session_id": session_id,
                "scenario": followup_q,
                "step": "followup",
                "scenario_type": scenario_type
            }
        else:
            # Move to next scenario or finish
            if scenario_idx + 1 < 7:  # Increased to 7 scenarios
                next_scenario, next_type = await generate_sales_scenario(role_type, buyer_profile)
                await collection.update_one(
                    {"_id": ObjectId(session_id)},
                    {"$set": {
                        "current_scenario": scenario_idx + 1,
                        "step": "scenario",
                        "last_scenario": next_scenario,
                        "scenario_type": next_type
                    }}
                )
                return {
                    "session_id": session_id,
                    "scenario": next_scenario,
                    "step": "scenario",
                    "scenario_type": next_type
                }
            else:
                # All done
                closing_prompt = (
                    "Thank you for participating in these sales scenarios. We appreciate your thoughtful responses.\n"
                    "If there's anything else you'd like to share about your approach to sales situations or how you handle complex scenarios, please let us know."
                )
                await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"step": "done"}})
                return {
                    "session_id": session_id,
                    "scenario": closing_prompt,
                    "step": "done",
                    "scenario_type": scenario_type
                }
    elif step == "followup":
        # After follow-up, move to next scenario or finish
        if scenario_idx + 1 < 7:  # Increased to 7 scenarios
            next_scenario, next_type = await generate_sales_scenario(role_type, buyer_profile)
            await collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {
                    "current_scenario": scenario_idx + 1,
                    "step": "scenario",
                    "last_scenario": next_scenario,
                    "scenario_type": next_type
                }}
            )
            return {
                "session_id": session_id,
                "scenario": next_scenario,
                "step": "scenario",
                "scenario_type": next_type
            }
        else:
            closing_prompt = (
                "Thank you for participating in these sales scenarios. We appreciate your thoughtful responses.\n"
                "If there's anything else you'd like to share about your approach to sales situations or how you handle complex scenarios, please let us know."
            )
            await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"step": "done"}})
            return {
                "session_id": session_id,
                "scenario": closing_prompt,
                "step": "done",
                "scenario_type": scenario_type
            }

async def generate_sales_scenario(role_type: str, buyer_profile: str) -> tuple[str, str]:
    """
    Generate a realistic B2B sales scenario based on role type and buyer profile.
    The scenario will be presented as a direct client conversation.
    """
    try:
        # Define scenario types
        scenario_types = [
            "pricing_pressure",
            "stakeholder_conflict",
            "timeline_pressure",
            "competitor_threat",
            "scope_creep",
            "budget_constraint",
            "implementation_concern"
        ]
        
        # Select a random scenario type
        scenario_type = random.choice(scenario_types)
        
        prompt = f"""
        Generate a realistic B2B sales scenario for a {role_type} selling to {buyer_profile}.
        Write the scenario as if you are the client speaking directly to the sales person.
        The scenario should:
        1. Be under 100 words
        2. Sound exactly like how a real client would speak in a business meeting
        3. Include natural hesitations, concerns, or objections
        4. Be specific to the {scenario_type} situation
        5. Avoid any sales jargon or formal language
        6. Feel like a real conversation, not a script

        Scenario type: {scenario_type}

        Example of natural client speech:
        "Look, I like your solution, but our CFO is breathing down my neck about the budget. We've got three other vendors promising similar features at 30% less. I need something concrete to take back to her."

        Return ONLY the client's dialogue as plain text. Make it sound like a real client conversation.
        """
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a client in a business meeting. Write exactly how you would speak to a sales person, with natural concerns and hesitations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 200
                }
            ) as response:
                if response.status != 200:
                    return "Could not generate sales scenario. Please try again.", scenario_type
                    
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                return content, scenario_type
    except Exception as e:
        logger.error(f"Error generating sales scenario: {str(e)}")
        return "Could not generate sales scenario. Please try again.", "error"

async def generate_sales_followup(role_type: str, buyer_profile: str, scenario: str, user_response: str) -> str:
    """
    Generate a follow-up question based on the user's response to a sales scenario.
    """
    try:
        prompt = f"""
        Given the following sales scenario and candidate's response, generate ONE natural client follow-up that:
        1. Sounds exactly like how a real client would speak
        2. Shows natural skepticism or concern
        3. Pushes for more specific details
        4. Maintains a realistic business conversation tone

        Role Type: {role_type}
        Buyer Profile: {buyer_profile}
        Client's Previous Statement: {scenario}
        Sales Person's Response: {user_response}

        Return ONLY the client's natural follow-up as plain text.
        """
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a client in a business meeting. Write exactly how you would respond to a sales person, with natural skepticism and business concerns."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 200
                }
            ) as response:
                if response.status != 200:
                    return "Could not generate follow-up question. Please try again."
                    
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                return content
    except Exception as e:
        logger.error(f"Error generating sales follow-up: {str(e)}")
        return "Could not generate follow-up question. Please try again."

async def needs_followup_openai(scenario: str, user_response: str) -> bool:
    """
    Determine if a follow-up question is needed based on the user's response.
    """
    try:
        prompt = f"""
        Given the following sales scenario and candidate's response, evaluate if a follow-up question is needed because the response:
        1. Lacks specific details about their approach
        2. Doesn't address potential objections or complications
        3. Could benefit from more strategic thinking
        4. Needs clarification on their reasoning

        Scenario: {scenario}
        Candidate Response: {user_response}

        Return ONLY 'yes' if a follow-up is needed, or 'no' if the response is sufficiently detailed and strategic.
        """
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert sales trainer evaluating response quality."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 10
                }
            ) as response:
                if response.status != 200:
                    return False
                    
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip().lower()
                return content.startswith('y')
    except Exception as e:
        logger.error(f"Error evaluating need for follow-up: {str(e)}")
        return False

class SalesConversationEvaluation(BaseModel):
    structured_thinking: dict
    commercial_awareness: dict
    stakeholder_judgment: dict
    negotiation_mindset: dict
    clarity_and_persuasion: dict
    overall_score: float
    summary: str

@app.post("/evaluate-sales-conversation/{session_id}")
async def evaluate_sales_conversation(session_id: str):
    """
    Evaluate a sales conversation based on multiple dimensions including structured thinking,
    commercial awareness, stakeholder judgment, negotiation mindset, and clarity & persuasion.
    """
    logger.info(f"Evaluating sales conversation for session {session_id}")
    
    try:
        # Fetch session from database
        collection = db["sales_scenarios"]
        session = await collection.find_one({"_id": ObjectId(session_id)})
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Extract responses from session
        responses = session.get("responses", [])
        if not responses:
            raise HTTPException(status_code=400, detail="No responses found in session")
        
        # Prepare conversation for evaluation
        conversation = "\n".join([
            f"Scenario: {response.get('scenario', '')}\nResponse: {response.get('response', '')}"
            for response in responses if response.get('type') != 'final_thoughts'
        ])
        
        # Create evaluation prompt
        prompt = f"""
        Evaluate the following sales conversation based on these dimensions. Return a JSON object with scores (1-5) and feedback for each category.

        Sales Conversation:
        {conversation}

        Evaluate based on these categories:

        1. Structured Thinking (1-5):
           - Score 1: Jumps into solution without clarifying problem or trade-offs
           - Score 3: Some structure but could be more systematic
           - Score 5: Clearly breaks down situation logically, lays out problem, options, and trade-offs

        2. Commercial Awareness (1-5):
           - Score 1: Ignores consequences to revenue, pricing, or deal structure
           - Score 3: Shows some understanding of business impact
           - Score 5: Clearly mentions impact on quota, margin, deal velocity, or customer lifetime value

        3. Stakeholder Judgment (1-5):
           - Score 1: Focuses only on one stakeholder or misses obvious dynamics
           - Score 3: Considers some stakeholders but misses key players
           - Score 5: Considers both internal (manager, product, legal) and external (buyer, influencer, blocker) players

        4. Negotiation Mindset (1-5):
           - Score 1: Immediately offers discounts or concessions without resistance
           - Score 3: Some attempt to maintain value but could be stronger
           - Score 5: Anchors on value, avoids caving, suggests creative or conditional levers

        5. Clarity & Persuasion (1-5):
           - Score 1: Rambling, unsure, or unable to explain trade-offs
           - Score 3: Clear enough but could be more compelling
           - Score 5: Communicates decisions clearly and with confidence; can justify approach to others

        Return a JSON object in this format:
        {{
            "structured_thinking": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "commercial_awareness": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "stakeholder_judgment": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "negotiation_mindset": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "clarity_and_persuasion": {{
                "score": 1-5,
                "feedback": "string"
            }},
            "overall_score": float,
            "summary": "string"
        }}
        """
        
        # Call Azure OpenAI for evaluation
        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert sales trainer evaluating sales conversations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "response_format": {"type": "json_object"}
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Failed to evaluate sales conversation")
                
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    evaluation = json.loads(content)
                    
                    # Validate scores are within range
                    for category in ["structured_thinking", "commercial_awareness", "stakeholder_judgment", 
                                   "negotiation_mindset", "clarity_and_persuasion"]:
                        if not 1 <= evaluation[category]["score"] <= 5:
                            evaluation[category]["score"] = max(1, min(5, evaluation[category]["score"]))
                    
                    # Calculate overall score as average of category scores
                    category_scores = [
                        evaluation["structured_thinking"]["score"],
                        evaluation["commercial_awareness"]["score"],
                        evaluation["stakeholder_judgment"]["score"],
                        evaluation["negotiation_mindset"]["score"],
                        evaluation["clarity_and_persuasion"]["score"]
                    ]
                    evaluation["overall_score"] = round(sum(category_scores) / len(category_scores), 2)
                    
                    # Save evaluation to database
                    await collection.update_one(
                        {"_id": ObjectId(session_id)},
                        {"$set": {"sales_conversation_evaluation": evaluation}}
                    )
                    
                    return SalesConversationEvaluation(**evaluation)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing evaluation response: {str(e)}")
                    raise HTTPException(status_code=500, detail="Failed to parse evaluation response")
                
    except Exception as e:
        error_msg = f"Error evaluating sales conversation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class VideoUploadResponse(BaseModel):
    # video_url: str
    # video_id: str
    message: str

# async def upload_to_blob_storage(file: UploadFile, user_id: str) -> tuple[str, str]:
#     """
#     Upload a video file to Azure Blob Storage and return the URL and blob name.
#     """
#     try:
#         # Generate a unique blob name
#         timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#         file_extension = os.path.splitext(file.filename)[1]
#         blob_name = f"interview-video-{user_id}-{timestamp}{file_extension}"
        
#         # Create blob service client
#         blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        
#         # Get container client
#         container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)
        
#         # Create container if it doesn't exist
#         try:
#             await container_client.create_container()
#         except ResourceExistsError:
#             pass
        
#         # Get blob client
#         blob_client = container_client.get_blob_client(blob_name)
        
#         # Read file content
#         content = await file.read()
        
#         # Upload the file
#         await blob_client.upload_blob(content, overwrite=True)
        
#         # Get the URL
#         video_url = blob_client.url
        
#         return video_url, blob_name
#     except Exception as e:
#         logger.error(f"Error uploading video to blob storage: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error uploading video: {str(e)}")
async def upload_to_blob_storage(file: UploadFile, user_id: str, reset_count: int) -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    blob_name = f"{user_id}_video_{reset_count}{file_extension}"

    blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_VIDEO_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(settings.AZURE_VIDEO_STORAGE_CONTAINER_NAME)

    try:
        await container_client.create_container()
    except ResourceExistsError:
        pass

    blob_client = container_client.get_blob_client(blob_name)

    # ⏱️ Measure chunked upload time
    start = time.perf_counter()
    await blob_client.upload_blob(
        data=file.file,
        blob_type="BlockBlob",
        overwrite=True,
        max_concurrency=4,  # You can experiment with 4, 8, etc.
        content_settings=ContentSettings(content_type=file.content_type)
    )
    duration = time.perf_counter() - start
    logger.info(f"Azure chunked upload time: {duration:.2f} seconds")

    return blob_client.url, blob_name

@app.post("/test-upload-speed")
async def test_upload(file: UploadFile = File(...)):
    start = time.perf_counter()
    await file.read()  # simulate reading only
    duration = time.perf_counter() - start
    return {"read_time": duration}
def validate_video_file(file: UploadFile) -> None:
    """
    Validate if the uploaded file is a video file.
    Checks both content type and file extension.
    """
    # Check content type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    # Check file extension
    allowed_extensions = {'.mp4', '.mov', '.avi', '.wmv', '.flv', '.mkv', '.webm'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid video format. Allowed formats: {', '.join(allowed_extensions)}"
        )

def validate_audio_file(file: UploadFile) -> None:
    """
    Validate if the uploaded file is an audio file.
    Checks both content type and file extension.
    """
    # Check content type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Only audio files are allowed")
    
    # Check file extension
    allowed_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid audio format. Allowed formats: {', '.join(allowed_extensions)}"
        )

@app.post("/upload-video/", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a video file to Azure Blob Storage and store the URL in the database.
    Also updates the user's resume profile with the video URL.
    """
    logger.info(f"Received video upload request for user: {user_id}")
    
    # Validate file type
    validate_video_file(file)
    
    try:
        # Check if user exists
        resume_collection = db["resume_profiles"]
        user_profile = await resume_collection.find_one({"_id": ObjectId(user_id)})
        if not user_profile:
            return JSONResponse(status_code=200, content={"message": "User profile not found"})
        
        reset_count = user_profile.get("video_interview_reset_count", 0)
        # Upload to blob storage
        video_url, blob_name = await upload_to_blob_storage(file, user_id, reset_count)
        processed_video_url= f'https://scootervideoconsumption.blob.core.windows.net/scooter-processed-videos/{user_id}_video_{reset_count}_master.m3u8'
        # Store video information in database
        video_doc = {
            "user_id": user_id,
            "video_url": video_url,
            "processed_video_url": processed_video_url,
            "blob_name": blob_name,
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": datetime.now(timezone.utc)
        }
        
        collection = db["videos"]
        result = await collection.insert_one(video_doc)
        
        # Update user's resume profile with video URL
        await resume_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "video_url": video_url,
                "processed_video_url": processed_video_url,
                "video_uploaded_at": datetime.now(timezone.utc)
            }}
        )
        
        return VideoUploadResponse(
            message="Video uploaded successfully"
            # video_url=video_url,
            # video_id=str(result.inserted_id)
        )
    except Exception as e:
        error_msg = f"Error processing video upload: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload-short-video/", response_model=VideoUploadResponse)
async def upload_short_video(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload a video file to Azure Blob Storage and store the URL in the database.
    Also updates the user's resume profile with the video URL.
    """
    logger.info(f"Received video upload request for user: {user_id}")
    
    # Validate file type
    validate_video_file(file)
    
    try:
        # Check if user exists
        resume_collection = db["resume_profiles"]
        user_profile = await resume_collection.find_one({"_id": ObjectId(user_id)})
        if not user_profile:
            return JSONResponse(status_code=200, content={"message": "User profile not found"})
        
        # Upload to blob storage
        video_url, blob_name = await upload_to_blob_storage(file, user_id)
        
        # Store video information in database
        video_doc = {
            "user_id": user_id,
            "short_video_url": video_url,
            "blob_name": blob_name,
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": datetime.now(timezone.utc)
        }
        
        collection = db["videos"]
        result = await collection.insert_one(video_doc)
        
        # Update user's resume profile with video URL
        await resume_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "short_video_url": video_url,
                "video_uploaded_at": datetime.now(timezone.utc)
            }}
        )
        
        return VideoUploadResponse(
            message="Video uploaded successfully"
            # video_url=video_url,
            # video_id=str(result.inserted_id)
        )
    except Exception as e:
        error_msg = f"Error processing video upload: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


class AudioUploadResponse(BaseModel):
    # audio_url: str
    # audio_id: str
    message: str
async def upload_to_blob_storage_audio(file: UploadFile, user_id: str) -> tuple[str, str]:
    """
    Upload an audio file to Azure Blob Storage and return the URL and blob name.
    """
    try:
        # Generate a unique blob name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        blob_name = f"interview-audio-{user_id}-{timestamp}{file_extension}"
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        
        # Get container client
        container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_AUDIO_CONTAINER_NAME)
        
        # Create container if it doesn't exist
        try:
            await container_client.create_container()
        except ResourceExistsError:
            pass
        
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Read file content
        content = await file.read()
        
        # Upload the file
        await blob_client.upload_blob(content, overwrite=True)
        
        # Get the URL
        audio_url = blob_client.url
        
        return audio_url, blob_name
    except Exception as e:
        logger.error(f"Error uploading audio to blob storage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading audio: {str(e)}")

@app.post("/upload-audio/", response_model=AudioUploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Upload an audio file to Azure Blob Storage and store the URL in the database.
    Also updates the user's resume profile with the audio URL.
    """
    logger.info(f"Received audio upload request for user: {user_id}")
    
    # Validate file type
    validate_audio_file(file)
    
    try:
        # Check if user exists
        resume_collection = db["resume_profiles"]
        user_profile = await resume_collection.find_one({"_id": ObjectId(user_id)})
        if not user_profile:
            return JSONResponse(status_code=200, content={"message": "User profile not found"})
        
        # Upload to blob storage
        audio_url, blob_name = await upload_to_blob_storage_audio(file, user_id)
        
        # Store audio information in database
        audio_doc = {
            "user_id": user_id,
            "audio_url": audio_url,
            "blob_name": blob_name,
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded_at": datetime.utcnow()
        }
        
        collection = db["audio_files"]
        result = await collection.insert_one(audio_doc)
        
        # Update user's resume profile with audio URL
        await resume_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "audio_url": audio_url,
                "audio_uploaded_at": datetime.utcnow()
            }}
        )
        
        return AudioUploadResponse(
            message="Audio uploaded successfully"
            # audio_url=audio_url,
            # audio_id=str(result.inserted_id)
        )
    except Exception as e:
        error_msg = f"Error processing audio upload: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class OptionalBasicInformation(BaseModel):
    full_name: Optional[str] = None
    current_location: Optional[str] = None
    open_to_relocation: Optional[bool] = None
    phone_number: Optional[str] = None
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    specific_phone_number: Optional[str] = None
    notice_period: Optional[str] = None
    current_ctc: Optional[dict] = None
    expected_ctc: Optional[dict] = None

class OptionalCareerOverview(BaseModel):
    total_years_experience: Optional[float] = None
    years_sales_experience: Optional[float] = None
    average_tenure_per_role: Optional[float] = None
    employment_gaps: Optional[EmploymentGaps] = None
    promotion_history: Optional[bool] = None
    company_history: Optional[List[CompanyHistory]] = None

class OptionalSalesContext(BaseModel):
    sales_type: Optional[Union[str, List[str]]] = None
    sales_motion: Optional[Union[str, List[str]]] = None
    industries_sold_into: Optional[List[str]] = None
    regions_sold_into: Optional[List[str]] = None
    buyer_personas: Optional[List[str]] = None

class OptionalRoleProcessExposure(BaseModel):
    sales_role_type: Optional[str] = None
    position_level: Optional[str] = None
    sales_stages_owned: Optional[List[str]] = None
    average_deal_size: Optional[str] = None
    sales_cycle_length: Optional[str] = None
    own_quota: Optional[bool] = None
    quota_ownership: Optional[List[str]] = None
    quota_attainment: Optional[str] = None

class OptionalToolsPlatforms(BaseModel):
    crm_tools: Optional[List[str]] = None
    crm_used: Optional[List[str]] = None  # Alternative field name
    sales_tools: Optional[List[str]] = None

class ProfileSearchRequest(BaseModel):
    basic_information: Optional[OptionalBasicInformation] = None
    career_overview: Optional[OptionalCareerOverview] = None
    sales_context: Optional[OptionalSalesContext] = None
    role_process_exposure: Optional[OptionalRoleProcessExposure] = None
    tools_platforms: Optional[OptionalToolsPlatforms] = None
    job_id: Optional[str] = None

@app.post("/search-profiles/")
async def search_profiles(
    search_request: ProfileSearchRequest,
    exact: bool = Query(False, description="If true, requires exact matches for all fields"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page")
):
    """
    Search for profiles that match the given criteria.
    Returns profiles ranked by their match score, including any existing evaluation scores.
    
    Parameters:
    - exact: If true, requires exact matches for all fields. If false, allows partial matches.
    - job_id: Optional job ID to filter profiles by specific job role
    - page: Page number (default: 1)
    - page_size: Number of items per page (default: 10, max: 100)
    """
    logger.info(f"Received profile search request (exact={exact}, job_id={search_request.job_id}, page={page}, page_size={page_size})")
    
    try:
        collection = db["resume_profiles"]
        interview_collection = db["interview_sessions"]
        sales_collection = db["sales_scenarios"]
        
        # Helper function to safely convert field to array for set operations
        def make_array_safe(field_path):
            return {
                "$cond": {
                    "if": {"$isArray": field_path},
                    "then": field_path,
                    "else": {"$cond": {
                        "if": {"$eq": [field_path, None]},
                        "then": [],
                        "else": [field_path]
                    }}
                }
            }
        
        # Helper function to create array intersection scoring
        def create_array_intersection_score(db_field, search_value, weight=2):
            if not search_value:  # If search value is empty, return 0
                return 0
                
            search_array = search_value if isinstance(search_value, list) else [search_value]
            
            return {
                "$cond": {
                    "if": {"$ne": [search_array, []]},
                    "then": {
                        "$multiply": [
                            weight,
                            {
                                "$size": {
                                    "$setIntersection": [
                                        make_array_safe({"$ifNull": [db_field, []]}),
                                        search_array
                                    ]
                                }
                            }
                        ]
                    },
                    "else": 0
                }
            }
        
        # Helper function for exact match scoring
        def create_exact_match_score(db_field, search_value, weight=2):
            if search_value is None:
                return 0
            
            return {
                "$cond": {
                    "if": {"$eq": [{"$ifNull": [db_field, ""]}, search_value]},
                    "then": weight,
                    "else": 0
                }
            }
        
        # Helper function for numeric comparison scoring
        def create_numeric_comparison_score(db_field, search_value, weight=3, is_exact=False):
            if search_value is None:
                return 0
            
            if is_exact:
                return {
                    "$cond": {
                        "if": {"$eq": [{"$ifNull": [db_field, 0]}, search_value]},
                        "then": weight,
                        "else": 0
                    }
                }
            else:
                return {
                    "$cond": {
                        "if": {"$gte": [{"$ifNull": [db_field, 0]}, search_value]},
                        "then": weight,
                        "else": 0
                    }
                }
        
        # Build match scores based on search criteria
        match_score_components = []
        
        # Basic Information matches
        if search_request.basic_information:
            bi = search_request.basic_information
            
            if bi.current_location:
                match_score_components.append(
                    create_exact_match_score("$basic_information.current_location", bi.current_location, 3)
                )
            
            if bi.open_to_relocation is not None:
                match_score_components.append(
                    create_exact_match_score("$basic_information.open_to_relocation", bi.open_to_relocation, 2)
                )
        
        # Career Overview matches
        if search_request.career_overview:
            co = search_request.career_overview
            
            if co.total_years_experience is not None:
                match_score_components.append(
                    create_numeric_comparison_score("$career_overview.total_years_experience", co.total_years_experience, 3, exact)
                )
            
            if co.years_sales_experience is not None:
                match_score_components.append(
                    create_numeric_comparison_score("$career_overview.years_sales_experience", co.years_sales_experience, 3, exact)
                )
        
        # Sales Context matches
        if search_request.sales_context:
            sc = search_request.sales_context
            
            if sc.sales_type:
                if exact:
                    # For exact match, convert both to arrays and check if they're equal
                    search_array = sc.sales_type if isinstance(sc.sales_type, list) else [sc.sales_type]
                    match_score_components.append({
                        "$cond": {
                            "if": {
                                "$eq": [
                                    make_array_safe({"$ifNull": ["$sales_context.sales_type", []]}),
                                    search_array
                                ]
                            },
                            "then": 2,
                            "else": 0
                        }
                    })
                else:
                    match_score_components.append(
                        create_array_intersection_score("$sales_context.sales_type", sc.sales_type, 2)
                    )
            
            if sc.sales_motion:
                if exact:
                    search_array = sc.sales_motion if isinstance(sc.sales_motion, list) else [sc.sales_motion]
                    match_score_components.append({
                        "$cond": {
                            "if": {
                                "$eq": [
                                    make_array_safe({"$ifNull": ["$sales_context.sales_motion", []]}),
                                    search_array
                                ]
                            },
                            "then": 2,
                            "else": 0
                        }
                    })
                else:
                    match_score_components.append(
                        create_array_intersection_score("$sales_context.sales_motion", sc.sales_motion, 2)
                    )
            
            if sc.industries_sold_into:
                if exact:
                    match_score_components.append({
                        "$cond": {
                            "if": {
                                "$eq": [
                                    make_array_safe({"$ifNull": ["$sales_context.industries_sold_into", []]}),
                                    sc.industries_sold_into
                                ]
                            },
                            "then": 2,
                            "else": 0
                        }
                    })
                else:
                    match_score_components.append(
                        create_array_intersection_score("$sales_context.industries_sold_into", sc.industries_sold_into, 2)
                    )
            
            if sc.regions_sold_into:
                if exact:
                    match_score_components.append({
                        "$cond": {
                            "if": {
                                "$eq": [
                                    make_array_safe({"$ifNull": ["$sales_context.regions_sold_into", []]}),
                                    sc.regions_sold_into
                                ]
                            },
                            "then": 2,
                            "else": 0
                        }
                    })
                else:
                    match_score_components.append(
                        create_array_intersection_score("$sales_context.regions_sold_into", sc.regions_sold_into, 2)
                    )
            
            if sc.buyer_personas:
                if exact:
                    match_score_components.append({
                        "$cond": {
                            "if": {
                                "$eq": [
                                    make_array_safe({"$ifNull": ["$sales_context.buyer_personas", []]}),
                                    sc.buyer_personas
                                ]
                            },
                            "then": 2,
                            "else": 0
                        }
                    })
                else:
                    match_score_components.append(
                        create_array_intersection_score("$sales_context.buyer_personas", sc.buyer_personas, 2)
                    )
        
        # Role Process Exposure matches
        if search_request.role_process_exposure:
            rpe = search_request.role_process_exposure
            
            if rpe.sales_role_type:
                match_score_components.append(
                    create_exact_match_score("$role_process_exposure.sales_role_type", rpe.sales_role_type, 3)
                )
            
            if rpe.position_level:
                match_score_components.append(
                    create_exact_match_score("$role_process_exposure.position_level", rpe.position_level, 3)
                )
        
        # Tools & Platforms matches
        if search_request.tools_platforms:
            tp = search_request.tools_platforms
            
            # Handle both crm_tools and crm_used fields
            crm_search_terms = []
            if tp.crm_tools:
                crm_search_terms.extend(tp.crm_tools if isinstance(tp.crm_tools, list) else [tp.crm_tools])
            if tp.crm_used:
                crm_search_terms.extend(tp.crm_used if isinstance(tp.crm_used, list) else [tp.crm_used])
            
            if crm_search_terms:
                if exact:
                    match_score_components.append({
                        "$cond": {
                            "if": {
                                "$eq": [
                                    make_array_safe({"$ifNull": ["$tools_platforms.crm_tools", []]}),
                                    crm_search_terms
                                ]
                            },
                            "then": 2,
                            "else": 0
                        }
                    })
                else:
                    match_score_components.append(
                        create_array_intersection_score("$tools_platforms.crm_tools", crm_search_terms, 2)
                    )
            
            if tp.sales_tools:
                if exact:
                    match_score_components.append({
                        "$cond": {
                            "if": {
                                "$eq": [
                                    make_array_safe({"$ifNull": ["$tools_platforms.sales_tools", []]}),
                                    tp.sales_tools
                                ]
                            },
                            "then": 2,
                            "else": 0
                        }
                    })
                else:
                    match_score_components.append(
                        create_array_intersection_score("$tools_platforms.sales_tools", tp.sales_tools, 2)
                    )
        
        # If no search criteria provided, return empty result
        if not match_score_components:
            return {
                "status": True,
                "message": "No search criteria provided",
                "total_profiles": 0,
                "exact_match": exact,
                "pagination": {
                    "current_page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "has_next": False,
                    "has_previous": False
                },
                "profiles": []
            }
        
        # Build the aggregation pipeline
        pipeline = [
            {
                "$match": {
                    "job_id": search_request.job_id if search_request.job_id else {"$exists": True}
                }
            },
            {
                "$addFields": {
                    "match_score": {"$add": match_score_components}
                }
            },
            {
                "$match": {
                    "match_score": {"$gt": 0}  # Only return profiles with some match
                }
            },
            {
                "$sort": {
                    "match_score": -1  # Sort by match score in descending order
                }
            }
        ]
        
        # Execute the aggregation to get total count
        count_pipeline = pipeline + [{"$count": "total"}]
        count_result = await collection.aggregate(count_pipeline).to_list(length=1)
        total_profiles = count_result[0]["total"] if count_result else 0
        
        # Calculate total pages
        total_pages = (total_profiles + page_size - 1) // page_size if total_profiles > 0 else 0
        
        # Validate page number
        if page > total_pages and total_pages > 0:
            raise HTTPException(status_code=400, detail=f"Page number exceeds total pages ({total_pages})")
        
        # Calculate skip value for pagination
        skip = (page - 1) * page_size
        
        # Add pagination to the pipeline
        paginated_pipeline = pipeline + [
            {"$skip": skip},
            {"$limit": page_size}
        ]
        
        # Execute the paginated aggregation
        results = await collection.aggregate(paginated_pipeline).to_list(length=page_size)
        
        # Convert ObjectId to string for JSON serialization and fetch scores
        for result in results:
            result["_id"] = str(result["_id"])
            if "created_at" in result:
                result["created_at"] = result["created_at"].isoformat()
            
            # Fetch communication evaluation scores
            interview_session = await interview_collection.find_one(
                {"user_id": result["_id"]},
                sort=[("created_at", -1)]  # Get the most recent session
            )
            
            if interview_session and "communication_evaluation" in interview_session:
                result["communication_scores"] = {
                    "content_and_thought": interview_session["communication_evaluation"]["content_and_thought"],
                    "verbal_delivery": interview_session["communication_evaluation"]["verbal_delivery"],
                    "non_verbal": interview_session["communication_evaluation"]["non_verbal"],
                    "presence_and_authenticity": interview_session["communication_evaluation"]["presence_and_authenticity"],
                    "overall_score": interview_session["communication_evaluation"]["overall_score"]
                }
            
            # Fetch sales conversation evaluation scores
            sales_session = await sales_collection.find_one(
                {"user_id": result["_id"]},
                sort=[("created_at", -1)]  # Get the most recent session
            )
            
            if sales_session and "sales_conversation_evaluation" in sales_session:
                result["sales_scores"] = {
                    "structured_thinking": sales_session["sales_conversation_evaluation"]["structured_thinking"],
                    "commercial_awareness": sales_session["sales_conversation_evaluation"]["commercial_awareness"],
                    "stakeholder_judgment": sales_session["sales_conversation_evaluation"]["stakeholder_judgment"],
                    "negotiation_mindset": sales_session["sales_conversation_evaluation"]["negotiation_mindset"],
                    "clarity_and_persuasion": sales_session["sales_conversation_evaluation"]["clarity_and_persuasion"],
                    "overall_score": sales_session["sales_conversation_evaluation"]["overall_score"]
                }
        
        # Calculate pagination metadata
        has_next = page < total_pages
        has_previous = page > 1
        
        return {
            "status": True,
            "message": "Profiles retrieved successfully",
            "total_profiles": total_profiles,
            "exact_match": exact,
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_previous
            },
            "profiles": results
        }
        
    except Exception as e:
        error_msg = f"Error searching profiles: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


# Global variables for chunked upload
block_lists_by_blob: Dict[str, List[BlobBlock]] = {}  # Map blob_name to its block list
MAX_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks

def generate_blob_name(user_id: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"interview-video-{user_id}-{timestamp}.webm"

class ChunkUploadRequest(BaseModel):
    block_id: str
    blob_name: str
    user_id: str

class FinalizeUploadRequest(BaseModel):
    blob_name: str
    user_id: str

@app.post("/upload-chunk/")
async def upload_chunk(
    file: UploadFile = File(...),
    block_id: str = Form(...),
    user_id: str = Form(...),
    blob_name: str = Form("")
):
    """
    Upload a chunk of video data to Azure Blob Storage.
    The chunk is staged as a block and its ID is added to the block list for the blob_name.
    If blob_name is not provided, generate a new one and return it to the client.
    """
    try:
        # Validate user exists
        resume_collection = db["resume_profiles"]
        user_profile = await resume_collection.find_one({"_id": ObjectId(user_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")

        # Validate block_id is base64 encoded
        try:
            decoded_block_id = base64.b64decode(block_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid block_id format. Must be base64 encoded.")

        # If blob_name is not provided, generate one
        if not blob_name:
            blob_name = generate_blob_name(user_id)

        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        
        # Get container client
        container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)
        
        # Create container if it doesn't exist
        try:
            await container_client.create_container()
        except ResourceExistsError:
            pass

        # Get blob client (root directory, not a subfolder)
        blob_client = container_client.get_blob_client(blob_name)

        # Read chunk data
        chunk_data = await file.read()
        
        # Stage the block
        await blob_client.stage_block(
            block_id=block_id,
            data=chunk_data,
            length=len(chunk_data)
        )

        # Initialize block list for blob if it doesn't exist
        if blob_name not in block_lists_by_blob:
            block_lists_by_blob[blob_name] = []

        # Add block to blob's block list
        block_lists_by_blob[blob_name].append(BlobBlock(block_id=block_id))

        return {
            "message": "Chunk uploaded successfully",
            "block_id": block_id,
            "chunk_size": len(chunk_data),
            "user_id": user_id,
            "blob_name": blob_name
        }

    except Exception as e:
        error_msg = f"Error uploading chunk: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/finalize-upload/")
async def finalize_upload(request: FinalizeUploadRequest):
    """
    Finalize the video upload by committing all staged blocks for the blob_name.
    """
    try:
        # Validate user exists
        resume_collection = db["resume_profiles"]
        user_profile = await resume_collection.find_one({"_id": ObjectId(request.user_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")

        # Check if blob_name has any blocks to commit
        if request.blob_name not in block_lists_by_blob or not block_lists_by_blob[request.blob_name]:
            raise HTTPException(status_code=400, detail="No blocks to commit for this blob_name")

        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
        
        # Get container client
        container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)
        
        # Get blob client (root directory)
        blob_client = container_client.get_blob_client(request.blob_name)

        # Commit the block list for this blob_name
        await blob_client.commit_block_list(block_lists_by_blob[request.blob_name])

        # Get the blob URL
        blob_url = blob_client.url

        # Update user's resume profile with video URL
        await resume_collection.update_one(
            {"_id": ObjectId(request.user_id)},
            {"$set": {
                "video_url": blob_url,
                "video_uploaded_at": datetime.utcnow(),
                "video_blob_name": request.blob_name
            }}
        )

        # Clear the block list for this blob_name
        block_lists_by_blob[request.blob_name] = []

        return {
            "message": "Upload finalized successfully",
            "video_url": blob_url,
            "user_id": request.user_id,
            "blob_name": request.blob_name
        }

    except Exception as e:
        error_msg = f"Error finalizing upload: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class CompanyProfile(BaseModel):
    company_name: str
    email: str
    contact_number: str
    description: str
    address: str
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters long")
    created_at: Optional[datetime] = None

class CompanySignupResponse(BaseModel):
    status: bool
    message: str
    company_id: str
    company_name: str
    email: str
    contact_number: str
    description: str
    address: str

@app.post("/company-signup/", response_model=CompanySignupResponse)
async def company_signup(profile: CompanyProfile):
    """
    Register a new company profile in the database.
    Returns the company ID upon successful creation.
    """
    logger.info(f"Received company signup request for: {profile.company_name}")
    
    try:
        # Add creation timestamp
        profile_dict = profile.dict()
        profile_dict["created_at"] = datetime.utcnow()
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, profile.email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Validate contact number format (basic validation)
        contact_pattern = r'^\+?[0-9\s-]{10,}$'
        if not re.match(contact_pattern, profile.contact_number):
            raise HTTPException(status_code=400, detail="Invalid contact number format")
        
        # Check if company email already exists
        collection = db["company_profiles"]
        existing_company = await collection.find_one({"email": profile.email})
        if existing_company:
            raise HTTPException(status_code=400, detail="Company with this email already exists")
        
        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(profile.password.encode('utf-8'), salt)
        profile_dict["password"] = hashed_password.decode('utf-8')
        
        # Insert into database
        result = await collection.insert_one(profile_dict)
        
        return CompanySignupResponse(
            status=True,
            message="Company profile created successfully",
            company_id=str(result.inserted_id),
            company_name=profile.company_name,
            email=profile.email,
            contact_number=profile.contact_number,
            description=profile.description,
            address=profile.address
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error creating company profile: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class JobBadge(BaseModel):
    label: str
    className: str

class JobRole(BaseModel):
    title: str
    description: str
    badges: List[str]
    company_id: str
    job_details: dict
    created_at: Optional[datetime] = None
    is_active: bool = True

class JobRoleResponse(BaseModel):
    status: bool
    message: str
    role_id: str

@app.post("/add-job-role/", response_model=JobRoleResponse)
async def add_job_role(role: JobRole):
    """
    Add a new job role for a company.
    Returns the role ID upon successful creation.
    """
    logger.info(f"Received job role creation request for company: {role.company_id}")
    
    try:
        # Validate company exists
        company_collection = db["company_profiles"]
        company = await company_collection.find_one({"_id": ObjectId(role.company_id)})
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        
        # Add creation timestamp
        role_dict = role.dict()
        role_dict["created_at"] = datetime.utcnow()
        
        # Insert into database
        collection = db["job_roles"]
        result = await collection.insert_one(role_dict)
        
        return JobRoleResponse(
            status=True,
            message="Job role created successfully",
            role_id=str(result.inserted_id)
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error creating job role: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/company-job-roles/{company_id}")
async def get_company_job_roles(company_id: str):
    """
    Get all active job roles for a specific company, along with application insights per role.
    """
    try:
        # Validate company exists
        company_collection = db["company_profiles"]
        company = await company_collection.find_one({"_id": ObjectId(company_id)})
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        
        # Get active job roles
        job_roles_collection = db["job_roles"]
        profile_collection = db["resume_profiles"]

        roles = await job_roles_collection.find({
            "company_id": company_id,
            "is_active": True
        }).to_list(length=None)

        result_roles = []
        for role in roles:
            role_id = str(role["_id"])

            # Query resume_profiles for this job role
            total_candidates = await profile_collection.count_documents({"job_id": role_id})
            audio_attended_count = await profile_collection.count_documents({
                "job_id": role_id,
                "audio_interview": {"$exists": True}
            })
            video_attended_count = await profile_collection.count_documents({
                "job_id": role_id,
                "video_interview_start": True
            })
            moved_to_video_round_count = await profile_collection.count_documents({
    "job_id": role_id,
    "$or": [
        { "video_email_sent": True },
        { "application_status": "SendVideoLink" }
    ]
})

            # Convert ObjectId and date for JSON
            role["_id"] = role_id
            if "created_at" in role:
                role["created_at"] = role["created_at"].isoformat()

            # Append counts
            role.update({
                "total_candidates": total_candidates,
                "audio_attended_count": audio_attended_count,
                "video_attended_count": video_attended_count,
                "moved_to_video_round_count": moved_to_video_round_count
            })

            result_roles.append(role)

        return {
            "status": True,
            "message": "Job roles retrieved successfully",
            "roles": result_roles
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error retrieving job roles: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class TimeRange(BaseModel):
    from_time: datetime | None = None
    to_time: datetime | None = None

@app.post("/ccompany-job-roles/{company_id}")
async def get_company_job_roles(company_id: str, time_range: TimeRange | None = Body(None)):
    try:
        company_collection = db["company_profiles"]
        company = await company_collection.find_one({"_id": ObjectId(company_id)})
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")

        job_roles_collection = db["job_roles"]
        profile_collection = db["resume_profiles"]

        roles = await job_roles_collection.find({
            "company_id": company_id,
            "is_active": True
        }).to_list(length=None)

        result_roles = []
        for role in roles:
            role_id = str(role["_id"])

            # Overall filters
            overall_filter = {"job_id": role_id}
            audio_overall_filter = {"job_id": role_id, "audio_interview": {"$exists": True}}
            video_overall_filter = {"job_id": role_id, "video_interview_start": True}
            moved_to_video_overall_filter = {
                "job_id": role_id,
                "$or": [{"video_email_sent": True}, {"application_status": "SendVideoLink"}]
            }

            # Overall counts
            total_candidates_overall = await profile_collection.count_documents(overall_filter)
            audio_attended_overall = await profile_collection.count_documents(audio_overall_filter)
            video_attended_overall = await profile_collection.count_documents(video_overall_filter)
            moved_to_video_overall = await profile_collection.count_documents(moved_to_video_overall_filter)

            # Timeframe counts
            timeframe_counts = None
            if time_range and time_range.from_time and time_range.to_time:
                created_time_filter = {
                    "job_id": role_id,
                    "created_at": {"$gte": time_range.from_time, "$lte": time_range.to_time}
                }
                audio_time_filter = {
                    "job_id": role_id,
                    "audio_interview": {"$exists": True},
                    "audio_uploaded_at": {"$gte": time_range.from_time, "$lte": time_range.to_time}
                }
                video_time_filter = {
                    "job_id": role_id,
                    "video_interview_start": True,
                    "video_uploaded_at": {"$gte": time_range.from_time, "$lte": time_range.to_time}
                }
                moved_to_video_time_filter = {
                "job_id": role_id,
                "$or": [
                    {
                        # Case 1: status updated in timeframe
                        "application_status_updated_at": {
                            "$gte": time_range.from_time,
                            "$lte": time_range.to_time
                        },
                        "$or": [
                            {"video_email_sent": True},
                            {"application_status": "SendVideoLink"}
                        ]
                    },
                    {
                        # Case 2: email sent in timeframe
                        "video_email_sent_at": {
                            "$gte": time_range.from_time,
                            "$lte": time_range.to_time
                        },
                        "$or": [
                            {"video_email_sent": True},
                            {"application_status": "SendVideoLink"}
                        ]
                    }
                ]
}
                timeframe_counts = {
                    "total_candidates": await profile_collection.count_documents(created_time_filter),
                    "audio_attended": await profile_collection.count_documents(audio_time_filter),
                    "video_attended": await profile_collection.count_documents(video_time_filter),
                    "moved_to_video_round": await profile_collection.count_documents(moved_to_video_time_filter)
                }

            # Final role update
            role["_id"] = role_id
            if "created_at" in role:
                role["created_at"] = role["created_at"].isoformat()

            role.update({
                "overall": {
                    "total_candidates": total_candidates_overall,
                    "audio_attended": audio_attended_overall,
                    "video_attended": video_attended_overall,
                    "moved_to_video_round": moved_to_video_overall
                },
                "timeframe": timeframe_counts  # will be dict or None -> JSON shows null
            })

            result_roles.append(role)

        return {
            "status": True,
            "message": "Job roles retrieved successfully",
            "roles": result_roles
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error retrieving job roles: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/all-jobs")
async def get_all_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    search: Optional[str] = Query(None, description="Search term for job title or description")
):
    """
    Get all active jobs from all companies with their company details.
    Supports pagination and search functionality.
    """
    logger.info(f"Fetching all jobs (page {page}, size {page_size})")
    
    try:
        # Get collections
        job_collection = db["job_roles"]
        company_collection = db["company_profiles"]
        
        # Build filter conditions
        filter_conditions = {}
        
        # Add search condition if provided
        if search:
            filter_conditions["$or"] = [
                {"title": {"$regex": search, "$options": "i"}},
                {"description": {"$regex": search, "$options": "i"}}
            ]
        
        # Get total count of jobs
        total_jobs = await job_collection.count_documents(filter_conditions)
        logger.info(f"Total jobs found: {total_jobs}")
        
        # Calculate total pages
        total_pages = (total_jobs + page_size - 1) // page_size
        
        # Validate page number
        if page > total_pages and total_pages > 0:
            raise HTTPException(status_code=400, detail=f"Page number exceeds total pages ({total_pages})")
        
        # Calculate skip value for pagination
        skip = (page - 1) * page_size
        
        # Find jobs with pagination
        jobs = await job_collection.find(
            filter_conditions
        ).sort([
    ("is_active", -1),   # active first
    ("created_at", -1)   # newest first
]).skip(skip).limit(page_size).to_list(length=page_size)
        
        logger.info(f"Found {len(jobs)} jobs after pagination")
        
        # Process each job to add company details
        jobs_with_company = []
        for job in jobs:
            # Get company details
            company = await company_collection.find_one({"_id": ObjectId(job["company_id"])})
            
            if company:
                job_with_company = {
                    "job_id": str(job["_id"]),
                    "title": job["title"],
                    "description": job["description"],
                    "badges": job["badges"],
                    "created_at": job["created_at"].isoformat() if "created_at" in job else None,
                    "is_active":job['is_active'],   
                    "company": {
                        "company_id": str(company["_id"]),
                        "company_name": company["company_name"],
                        "email": company["email"],
                        "contact_number": company["contact_number"],
                        "description": company["description"],
                        "address": company["address"]
                    }
                }
                jobs_with_company.append(job_with_company)
        
        # Calculate pagination metadata
        has_next = page < total_pages
        has_previous = page > 1
        
        response = {
            "status": True,
            "message": "Jobs retrieved successfully",
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_jobs": total_jobs,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_previous
            },
            "jobs": jobs_with_company
        }
        
        logger.info(f"Response prepared with {len(jobs_with_company)} jobs")
        return response
        
    except Exception as e:
        error_msg = f"Error retrieving jobs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/job-candidates/{job_id}")
async def get_job_candidates(
    job_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    audio_attended: Optional[bool] = Query(None, description="Filter by audio interview status"),
    video_attended: Optional[bool] = Query(None, description="Filter by video upload status"),
    video_interview_sent: Optional[bool] = Query(None, description="Filter by video interview sent status"),
    application_status: Optional[str] = Query(None, description="Filter by application status"),
    shortlisted: Optional[bool] = Query(None, description="Filter by shortlisted status"),
    all_candidates: Optional[bool] = Query(False, description="If true, ignore other filters and return all candidates"),
    call_for_interview: Optional[bool] = Query(None, description="Filter by call for interview status")
):
    """
    Get candidates for a specific job role with optional filters.
    """
    try:
        # Validate job role exists
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job role not found")
        # if not job.get("is_active", True):
        #     raise HTTPException(status_code=400, detail="Inactive job role")
        
        # Get all profiles for this job role
        profile_collection = db["resume_profiles"]
        interview_collection = db["interview_sessions"]
        sales_collection = db["sales_scenarios"]
        audio_collection= db["audio_interview_results"]
        audio_proctoring_collection = db["audio_proctoring_logs"]
        video_proctoring_collection = db["video_proctoring_logs"]
        
        # Build filter conditions
        filter_conditions = {"job_id": job_id}
        
        if all_candidates is not None:
            if all_candidates:
                filter_conditions = {"temp_list": True}

        if audio_attended is not None:
            if audio_attended:
                filter_conditions["audio_interview"] = True
            else:
                filter_conditions["$or"] = [
                    {"audio_interview": False},
                    {"audio_interview": {"$exists": False}}
                ]
        
        if application_status is not None:
            if application_status == "SendVideoLink":
                filter_conditions["$or"] = [
                    { "video_email_sent": True },
                    { "application_status": "SendVideoLink" }
                ]
            else:
                filter_conditions["application_status"] = application_status
        if video_attended is not None:
            if video_attended:
                filter_conditions["video_interview_start"] = True
            else:
                filter_conditions["$or"] = [
                    {"video_interview_start": False},
                    {"video_interview_start": {"$exists": False}}
                ]
        if video_interview_sent is not None:
            if video_interview_sent:
                filter_conditions["video_email_sent"] = True
            else:
                filter_conditions["$or"] = [
                    {"video_email_sent": False},
                    {"video_email_sent": {"$exists": False}}
                ]

        if shortlisted is not None:
            if shortlisted:
                filter_conditions["final_shortlist"] = True
            else:
                filter_conditions["$or"] = [
                    {"final_shortlist": False},
                    {"final_shortlist": {"$exists": False}}
                ]
        if call_for_interview is not None:
            if call_for_interview:
                filter_conditions["call_for_interview"] = True
            else:
                filter_conditions["call_for_interview"] = False
        # Log the filter conditions for debugging
        logger.info(f"Filter conditions: {filter_conditions}")
        
        # Get total count of candidates with filters
        total_candidates = await profile_collection.count_documents(filter_conditions)
        audio_attended_count = await profile_collection.count_documents({
            **filter_conditions,
            "audio_interview": True
        })
        # Get video attended count with filters
        video_attended_count = await profile_collection.count_documents({
            **filter_conditions,
            "video_interview_start": True
        })
        moved_to_video_round_count = await profile_collection.count_documents({
    **filter_conditions,
    "job_id": job_id,
    "$or": [
        { "video_email_sent": True },
        { "application_status": "SendVideoLink" }
    ]
})

        logger.info(f"Total candidates found: {total_candidates}")
        
        # Calculate total pages
        total_pages = (total_candidates + page_size - 1) // page_size
        
        # Validate page number
        if page > total_pages and total_pages > 0:
            raise HTTPException(status_code=400, detail=f"Page number exceeds total pages ({total_pages})")
        
        # Calculate skip value for pagination
        skip = (page - 1) * page_size
        
        # Find profiles with pagination and filters
        profiles = await profile_collection.find(
            filter_conditions
        ).sort("created_at", -1).skip(skip).limit(page_size).to_list(length=page_size)
        
        logger.info(f"Found {len(profiles)} profiles after pagination")
        
        # Process each profile to add interview details
        candidates = []
        for profile in profiles:
            career_overview = profile.get("career_overview", {})
            company_history = career_overview.get("company_history", [])

            # Sort by latest start_date first
            if company_history:
                company_history.sort(key=lambda x: x.get("start_date", ""), reverse=True)

            total_months = 0
            now = datetime.utcnow()

            for role in company_history:
                start_date_str = role.get("start_date")
                end_date_str = role.get("end_date")
                is_current = role.get("is_current", False)

                # Parse start_date
                try:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
                except ValueError:
                    start_date = None

                # If current, calculate till now
                if is_current and start_date:
                    diff = (now.year - start_date.year) * 12 + (now.month - start_date.month)
                    role["duration_months"] = diff
                    total_months += diff
                elif start_date and end_date_str:
                    # If both dates are present, calculate duration
                    try:
                        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                        diff = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                        role["duration_months"] = diff
                        total_months += diff
                    except ValueError:
                        pass  # skip if end_date is malformed
                else:
                    # Fall back to existing duration if no dates
                    total_months += role.get("duration_months", 0)

            total_years_experience = round(total_months / 12, 1)
            career_overview["company_history"] = company_history
            career_overview["total_years_experience"] = total_years_experience
            candidate = {
                "profile_id": str(profile["_id"]),
                "profile_created_at": profile["created_at"].isoformat() if "created_at" in profile else None,
                "short_summary": profile.get("short_summary", ""),
                "job_fit_assessment": profile.get("job_fit_assessment", " "),
                "basic_information": profile.get("basic_information", {}),
                "application_status": profile.get("application_status", ""),
                "application_status_reason": profile.get("application_status_reason", ""),
                "application_status_updated_at": profile.get("application_status_updated_at", None),
                "candidate_source": profile.get("candidate_source", ""),
                "final_shortlist": profile.get("final_shortlist", False),
                "shortlist_status_reason": profile.get("shortlist_status_reason", ""),
                "call_for_interview": profile.get("call_for_interview", False),
                "call_for_interview_notes": profile.get("call_for_interview_notes", ""),
                "career_overview": career_overview,
                "interview_status": {
                    "audio_interview_passed": profile.get("audio_interview", False),
                    "video_interview_attended": bool(profile.get("video_url")),
                    "audio_interview_attended": bool(profile.get("audio_url")),
                    "video_email_sent": profile.get("video_email_sent", False),
                    "video_interview_url": profile.get("video_url") if profile.get("video_url") else None,
                    "processed_video_interview_url": profile.get("processed_video_url") if profile.get("processed_video_url") else None,
                    "audio_interview_url": profile.get("audio_url") if profile.get("audio_url") else None,
                    "resume_url": profile.get("resume_url") if profile.get("resume_url") else None
                }
            }
            
            # Get latest interview session
            interview_session = await interview_collection.find_one(
                {"user_id": str(profile["_id"])},
                sort=[("created_at", -1)]
            )
            
            if interview_session:
                candidate["interview_details"] = {
                    "session_id": str(interview_session["_id"]),
                    "created_at": interview_session["created_at"].isoformat(),
                    "communication_evaluation": interview_session.get("communication_evaluation", {}),
                    "qa_evaluations": interview_session.get("evaluation", [])
                }
            audio_interview_session = await audio_collection.find_one(
                {"user_id": str(profile["_id"])},
                sort=[("created_at", -1)]
            )
            if audio_interview_session:
                candidate["audio_interview_details"]= {
                    "audio_interview_id": str(audio_interview_session["_id"]),
                    "created_at":audio_interview_session["created_at"].isoformat(),
                    "qa_evaluations": audio_interview_session.get("qa_evaluations",{}),
                    "audio_interview_summary": audio_interview_session.get("interview_summary",[])
                }
            
            audio_proctoring= await audio_proctoring_collection.find_one(
                {"user_id": str(profile["_id"])},
                sort=[("created_at", -1)]
                )
            if audio_proctoring:
                candidate["audio_proctoring_details"] = serialize_document(dict(audio_proctoring))

            video_proctoring = await video_proctoring_collection.find_one(
                {"user_id": str(profile["_id"])},
                sort=[("created_at", -1)]
            )
            if video_proctoring:
                candidate["video_proctoring_details"] = serialize_document(dict(video_proctoring))
            # Get latest sales scenario session
            sales_session = await sales_collection.find_one(
                {"user_id": str(profile["_id"])},
                sort=[("created_at", -1)]
            )
            
            if sales_session:
                candidate["sales_scenario_details"] = {
                    "session_id": str(sales_session["_id"]),
                    "created_at": sales_session["created_at"].isoformat(),
                    "sales_conversation_evaluation": sales_session.get("sales_conversation_evaluation", {}),
                    "responses": sales_session.get("responses", [])
                }
            
            candidates.append(candidate)
        
        # Calculate pagination metadata
        has_next = page < total_pages
        has_previous = page > 1
        
        response = {
            "status": True,
            "message": "Candidates retrieved successfully",
            "job_details": {
                "title": job["title"],
                "description": job["description"],
                "company_id": job["company_id"],
                "moved_to_video_round_count": moved_to_video_round_count,
                "audio_attended_count": audio_attended_count,
                "video_attended_count": video_attended_count,
                "candidate_count": total_candidates
            },
            "filters": {
                "audio_attended": audio_attended,
                "video_attended": video_attended,
                "application_status": application_status
            },
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_candidates": total_candidates,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_previous": has_previous
            },
            "candidates": candidates
        }
        
        logger.info(f"Response prepared with {len(candidates)} candidates")
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error retrieving job candidates: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class CompanyLoginRequest(BaseModel):
    email: str
    password: str

class CompanyLoginResponse(BaseModel):
    status: bool
    message: str
    company_id: str
    company_name: str
    email: str
    contact_number: str
    description: str
    address: str

@app.post("/company-login/", response_model=CompanyLoginResponse)
async def company_login(login_request: CompanyLoginRequest):
    """
    Authenticate a company using email and password.
    Returns company details upon successful login.
    """
    logger.info(f"Received company login request for email: {login_request.email}")
    
    try:
        # Find company by email
        collection = db["company_profiles"]
        company = await collection.find_one({"email": login_request.email})
        
        if not company:
            return JSONResponse(content={"status": False, "message": "Invalid email or password"}, status_code=200)
        
        # Verify password
        stored_password = company["password"].encode('utf-8')
        if not bcrypt.checkpw(login_request.password.encode('utf-8'), stored_password):
            return JSONResponse(content={"status": False, "message": "Invalid email or password"}, status_code=200)
        
        return CompanyLoginResponse(
            status=True,
            message="Login successful",
            company_id=str(company["_id"]),
            company_name=company["company_name"],
            email=company["email"],
            contact_number=company["contact_number"],
            description=company["description"],
            address=company["address"]
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error during company login: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class ApplicationStatusRequest(BaseModel):
    user_id: str
    application_status: str
    reason: Optional[str] = None

@app.post("/application-status/")
async def update_application_status(request: ApplicationStatusRequest):
    """
    Update the application status for a user in the resume profile collection.
    """
    try:
        collection = db["resume_profiles"]
        
        # Check if user exists
        user_profile = await collection.find_one({"_id": ObjectId(request.user_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Update the application status
        update_data = {
            "application_status": request.application_status,
            "application_status_updated_at": datetime.utcnow()
        }
        
        if request.reason:
            update_data["application_status_reason"] = request.reason
        
        await collection.update_one(
            {"_id": ObjectId(request.user_id)},
            {"$set": update_data}
        )
        
        return {
            "message": "Application status updated successfully",
            "user_id": request.user_id,
            "application_status": request.application_status,
            "reason": request.reason
        }
    except Exception as e:
        error_msg = f"Error updating application status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class VideoInterviewLoginRequest(BaseModel):
    email: str
    code: str

class VideoInterviewLoginResponse(BaseModel):
    status: bool
    message: str
    user_id: Optional[str] = None
    full_name: Optional[str] = None
    resume_status: bool
    reset_count: Optional[int] = None
    job_title: Optional[str] = None
    job_description: Optional[str] = None

@app.post("/video-interview-login/", response_model=VideoInterviewLoginResponse)
async def video_interview_login(login_request: VideoInterviewLoginRequest):
    """
    Validate email and OTP code for video interview access
    """
    try:
        collection = db["resume_profiles"]
        
        # Find all profiles with the given email
        cursor = collection.find(
            {"basic_information.email": login_request.email},
            sort=[("created_at", -1)]  # Sort by created_at in descending order
        )
        
        profiles = await cursor.to_list(length=None)
        
        if not profiles:
            return VideoInterviewLoginResponse(
                status=False,
                message="No profile found with this email address",
                resume_status=False

            )
            
        # Find the profile with matching OTP
        matching_profile = None
        for profile in profiles:
            if profile.get("interview_otp") == login_request.code:
                matching_profile = profile
                break
                
        if not matching_profile:
            return VideoInterviewLoginResponse(
                status=False,
                message="Invalid verification code",
                resume_status=False
            )
            
        # Check if the user has passed the audio interview
        # if not matching_profile.get("application_status", False):
        #     return VideoInterviewLoginResponse(
        #         status=False,
        #         message="You have not qualified for the video interview round",
        #         resume_status=False
        #     )
        if matching_profile.get("video_interview_start", False):
            return VideoInterviewLoginResponse(
                status=False,
                message="You have already attempted this interview",
                resume_status=False
            )
        if matching_profile.get("application_expired", False):
            return VideoInterviewLoginResponse(
                status=False,
                message="The last date to take this interview has expired",
                resume_status=False
            )
        # Fetch job title and description using job_id
        job_title = ""
        job_description = ""
        job_id = matching_profile.get("job_id")
        if job_id:
            try:
                job_obj_id = ObjectId(job_id)  # Remove 'J' prefix if added
                job_roles_collection = db["job_roles"]
                job_data = await job_roles_collection.find_one({"_id": job_obj_id})
                if job_data:
                    job_title = job_data.get("title", "")
                    job_description = job_data.get("description", "")
            except Exception as job_err:
                logger.warning(f"Job fetch failed for job_id: {job_id} - {str(job_err)}")
        return VideoInterviewLoginResponse(
            status=True,
            message="Login successful",
            user_id=str(matching_profile["_id"]),
            reset_count=matching_profile.get("video_interview_reset_count", 0),
            full_name=matching_profile.get("basic_information", {}).get("full_name", ""),
            resume_status=bool(matching_profile.get("resume_text", "").strip()),
            job_title=job_title,
            job_description=job_description
        )
        
    except Exception as e:
        error_msg = f"Error during video interview login: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/job/{job_id}")
async def get_job_details(job_id: str):
    """
    Get detailed information about a specific job role including company details.
    """
    try:
        # Get collections
        job_collection = db["job_roles"]
        company_collection = db["company_profiles"]
        
        # Find the job
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            return JSONResponse(status_code=200, content={"message": "Job not found"})
        
        # Get company details
        company = await company_collection.find_one({"_id": ObjectId(job["company_id"])})
        if not company:
            return JSONResponse(status_code=200, content={"message": "company not found"})
        
        # Format response
        response = {
            "job_id": str(job["_id"]),
            "title": job["title"],
            "description": job["description"],
            "badges": job["badges"],
            "created_at": job["created_at"].isoformat() if "created_at" in job else None,
            "company": {
                "company_id": str(company["_id"]),
                "company_name": company["company_name"],
                "email": company["email"],
                "contact_number": company["contact_number"],
                "description": company["description"],
                "address": company["address"]
            }
        }
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error retrieving job details: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class FinalShortlistRequest(BaseModel):
    user_id: str
    final_shortlist: bool
    reason: Optional[str] = None    
@app.post("/mark-final-shortlist/")
async def shortlist_application_status(request:FinalShortlistRequest):
    """
    Update the application status for a user in the resume profile collection.
    """
    try:
        collection = db["resume_profiles"]
        
        # Check if user exists
        user_profile = await collection.find_one({"_id": ObjectId(request.user_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Update the application status
        update_data = {
            "final_shortlist": request.final_shortlist,
            "shortlist_status_updated_at": datetime.utcnow()
        }
        
        if request.reason:
            update_data["shortlist_status_reason"] = request.reason
        
        await collection.update_one(
            {"_id": ObjectId(request.user_id)},
            {"$set": update_data}
        )
        
        return {
            "message": "Shortlist status updated successfully",
            "user_id": request.user_id,
            "shortlist_status": request.final_shortlist,
            "reason": request.reason
        }
    except Exception as e:
        error_msg = f"Error updating shortlist status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    
class CallForInterviewRequest(BaseModel):
    user_id: str
    call_for_interview: bool
    notes: Optional[str] = None
@app.post("/call-for-interview/")
async def update_application_status(request: CallForInterviewRequest):
    """
    Update the application status for a user in the resume profile collection.
    """
    try:
        collection = db["resume_profiles"]
        
        # Check if user exists
        user_profile = await collection.find_one({"_id": ObjectId(request.user_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Update the application status
        update_data = {
            "call_for_interview": request.call_for_interview,
            "call_for_interview_status_updated_at": datetime.utcnow()
        }
        
        if request.notes:
            update_data["call_for_interview_notes"] = request.notes
        
        await collection.update_one(
            {"_id": ObjectId(request.user_id)},
            {"$set": update_data}
        )
        
        return {
            "message": "Candidate interview status updated successfully",
            "user_id": request.user_id,
            "application_status": request.call_for_interview,
            "reason": request.notes
        }
    except Exception as e:
        error_msg = f"Error updating application status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
############################################################################################
# openai_client = AzureOpenAI(
#     api_key="9gE6yVCE4nGisb0Zf8yp15DMdl9qpN56z7bJ8AntLqOlX3TfP3EPJQQJ99BAACYeBjFXJ3w3AAABACOGQnpW",
#     azure_endpoint="https://imaigen-college-openai-service.openai.azure.com",
#     api_version="2024-08-01-preview",
#     azure_deployment="gpt-4o-mini"
# )
search_service_name = "scooter-knowledge"
admin_key = "yVz6bNG0y8ufcX6AN4ze6cwWz9lBA7MmgXqufcYGulAzSeDqNwDN"
index_name = "job-index"

search_client = SearchClient(
    endpoint=f"https://{search_service_name}.search.windows.net",
    index_name=index_name,
    credential=AzureKeyCredential(admin_key)
)

class Query(BaseModel):
    question: str
    job_id: str

async def async_search_azure(query: str, job_id: str) -> list[dict]:
    """Fully async Azure Search using REST API"""
    search_url = f"https://{search_service_name}.search.windows.net/indexes/{index_name}/docs/search?api-version=2023-11-01"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": admin_key
    }
    
    search_body = {
        "search": "*",
        "filter": f"id eq '{job_id}'",
        "top": 5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(search_url, headers=headers, json=search_body) as response:
                logger.info(f"Azure Search response status: {response}")
                if response.status == 200:
                    result = await response.json()
                    return result.get("value", [])
                else:
                    error_text = await response.text()
                    logger.error(f"Azure Search API error: {response.status} - {error_text}")
                    return []
    except Exception as e:
        logger.error(f"Azure search error: {e}")
        return []

@app.post("/ask-job")
async def ask_job_question_async(query: Query):
    try:
        # Fully async Azure Search
        search_results = await async_search_azure(query.question, query.job_id)
        logger.info(f"Search results: {search_results}")
        if not search_results:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get job title from search results
        job_title = search_results[0].get("title", "the role")

        context_docs = [
            doc.get("content")
            for doc in search_results
            if doc.get("content")
        ]

        context_text = "\n\n".join(context_docs[:5]) or "No additional job context was found."

        prompt = f"""You are scooty a helpful assistant that answers candidate questions about job roles.
The user is asking about the job: "{job_title}".
Use only the following job-specific information to answer:

{context_text}

Question: {query.question}
Answer:"""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(status_code=500, detail=f"OpenAI error: {response.status} - {error_text}")

                result = await response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                
                # Store the Q&A in chatbot collection
                try:
                    chatbot_collection = db["chatbot"]
                    chatbot_entry = {
                        "job_id": query.job_id,
                        "question": query.question,
                        "answer": answer,
                        "job_title": job_title,
                        "timestamp": datetime.utcnow(),
                        "search_results_count": len(search_results)
                    }
                    
                    await chatbot_collection.insert_one(chatbot_entry)
                    logger.info(f"Stored chatbot Q&A for job {query.job_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to store chatbot Q&A: {str(e)}")
                    # Don't fail the request if storage fails
                
                return {
                    "status": True,
                    "answer": answer
                }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in /ask-async")
        raise HTTPException(status_code=500, detail=str(e))

######################

# This function can be replaced by your existing Azure call
async def gen_interview_questions_from_resume(resume: str, posting_title: str) -> list[str]:
    prompt = f"""
You are an expert interviewer.

Using the resume below, generate 6 open-ended, conversational interview questions for a sales candidate applying to the job title: {posting_title}.

Follow this structure:

1. **Daily Reality**: Ask what the candidate usually did in the first few hours of the day in their most recent sales role (mention the company from the resume).
2. **Tools & Likes/Dislikes**: Ask which tools or systems they used at any two companies from their resume, and which they liked or disliked.
3. **Resume Claim Deep-Dive**: Choose one specific achievement or claim from the resume and ask them to explain how they achieved it.
4. **Common Pushback**: Ask what was the most common objection they heard from prospects and how they responded.
5. **Career Moves**: Include this only if the resume shows gaps >2 months, multiple <1-year stints, or clear pivots. Ask why they made those moves or had breaks. If no such cases, just ignore this.
6. **Self-Awareness**: Ask what sales skill they’d like to improve and how they’re working on it.

Candidate Resume:
{resume}

Only return a valid Python list of string questions. If question 5 is not applicable, skip it and return other questions.
No explanation. No markdown. No extra text.
"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional interviewer generating high-quality, role-relevant interview questions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 600
            }
        ) as response:
            if response.status != 200:
                logger.error(f"Azure OpenAI API returned status {response.status}")
                raise HTTPException(status_code=500, detail="Error calling LLM")

            result = await response.json()
            try:
                question_list = result["choices"][0]["message"]["content"].strip()
                return eval(question_list) if isinstance(question_list, str) else question_list
            except Exception as e:
                logger.error(f"Failed to parse questions: {e}, Response: {result}", exc_info=True)
                raise HTTPException(status_code=500, detail="Invalid response format from LLM")

async def gen_interview_questions(profile_id: str, posting_title: str) -> List[str]:
    logger.info(f"Generating interview questions for {posting_title} based on candidate summary")

    # Fetch resume profile from database
    collection = db["resume_profiles"]
    profile = await collection.find_one({"_id": ObjectId(profile_id)})

    if not profile:
        logger.error(f"Profile not found for ID: {profile_id}")
        return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)

    resume = profile.get("resume_text")
    if not resume or not resume.strip():
        logger.warning(f"No candidate summary available for profile {profile_id}")
        return JSONResponse(content={"message": "resume text not available"}, status_code=200)

    # Generate questions based on the summary
    questions = await gen_interview_questions_from_resume(resume, posting_title)
    return questions


@app.post("/generate-interview-questions/")
async def gen_questions(request: InterviewQuestionRequest):
    """
    Generate interview questions based on stored candidate summary and job title
    """
    logger.info(f"Received request to generate questions for {request.posting_title} position")
    try:
        collection = db["resume_profiles"]
        profile = await collection.find_one({"_id": ObjectId(request.profile_id)})

        if not profile:
            return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)
        if "interview_otp" in profile:
            return JSONResponse(content={"message": "Audio round already completed", "status": False}, status_code=200)

        questions = await gen_interview_questions(request.profile_id, request.posting_title)
        return {"questions": questions}

    except Exception as e:
        error_msg = f"Error generating questions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class AudioSummaryRequest(BaseModel):
    user_id: str

@app.post("/update-summary-from-audio")
async def update_summary_from_audio(request: AudioSummaryRequest):
    try:
        user_id = request.user_id

        # Fetch user resume profile
        profile_collection = db["resume_profiles"]
        user = await profile_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch audio evaluations
        audio_collection = db["audio_interview_results"]
        audio_doc = await audio_collection.find_one({"user_id": user_id})
        if not audio_doc or "interview_summary" not in audio_doc:
            raise HTTPException(status_code=404, detail="Audio interview evaluation not found")

        interview_summary = audio_doc["interview_summary"]
        avg_score = interview_summary.get("average_score", "N/A")
        strengths = interview_summary.get("strengths", [])
        improvement_areas = interview_summary.get("areas_for_improvement", [])

        # Construct input for LLM
        audio_context = f"""
The following is an evaluation summary from the candidate's audio interview round:

Average Score: {avg_score}
Strengths: {', '.join(strengths) if strengths else 'None mentioned'}
Areas for Improvement: {', '.join(improvement_areas)}

Using this, revise the candidate's professional summary to reflect both their resume and their spoken responses.
If confidence, communication, or clarity were poor, mention that constructively.
Start the summary with "Candidate's name is" and write 3–5 sentences.
Do not include any extra formatting or explanation.
"""

        # Combine with resume_text if available
        resume_text = user.get("resume_text", "").strip()
        combined_prompt = f"""You are an expert recruiter. Based on the resume and audio evaluation, generate a 3–5 sentence candidate summary.

Resume:
{resume_text}

Audio Evaluation:
{audio_context}
"""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a professional recruiter summarizing candidate profiles."},
                        {"role": "user", "content": combined_prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 250
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"Azure OpenAI API returned status {response.status}")
                    raise HTTPException(status_code=500, detail="Error calling LLM")

                result = await response.json()
                try:
                    summary = result["choices"][0]["message"]["content"].strip()

                    # Update the profile with the new summary
                    await profile_collection.update_one(
                        {"_id": ObjectId(user_id)},
                        {"$set": {"short_summary": summary}}
                    )

                    return {
                        "status": True,
                        "updated_candidate_summary": summary
                    }

                except (KeyError, IndexError):
                    raise HTTPException(status_code=500, detail="Invalid response format from LLM")

    except Exception as e:
        logger.error(f"Error updating summary from audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

async def update_candidate_summary_from_audio(user_id: str) -> str:
    collection = db["resume_profiles"]
    audio_collection = db["audio_interview_results"]
    
    user = await collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    audio_doc = await audio_collection.find_one({"user_id": user_id})
    if not audio_doc or "interview_summary" not in audio_doc:
        raise HTTPException(status_code=404, detail="Audio evaluation not found")

    interview_summary = audio_doc["interview_summary"]
    avg_score = interview_summary.get("average_score", "N/A")
    strengths = interview_summary.get("strengths", [])
    improvement_areas = interview_summary.get("areas_for_improvement", [])

    short_summary = user.get("short_summary", "").strip()

    prompt = f"""You are an expert recruiter. Based on the candidates's summary: and audio evaluation, generate a 5-7 sentence candidate summary.
Start the summary with the candidate's name if given. Do not include any formatting or extra explanation.

candidates's summary:
{short_summary}

Audio Evaluation Summary:
Average Score: {avg_score}
Strengths: {', '.join(strengths) or 'None mentioned'}
Areas for Improvement: {', '.join(improvement_areas) or 'None mentioned'}
"""

    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": [
                    {"role": "system", "content": "You are a professional recruiter summarizing candidate profiles."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 250
            }
        ) as response:
            if response.status != 200:
                logger.error(f"Azure OpenAI API failed with status {response.status}")
                raise HTTPException(status_code=500, detail="LLM call failed")

            result = await response.json()
            summary = result["choices"][0]["message"]["content"].strip()
            await collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"short_summary": summary}}
            )
            return summary

async def update_candidate_summary_from_video(user_id: str) -> str:
    logger.info(f"Starting candidate summary update from video for user_id: {user_id}")
    
    collection = db["resume_profiles"]
    video_collection = db["interview_sessions"]

    try:
        user = await collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            logger.warning(f"User not found in resume_profiles for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="User not found")

        video_doc = await video_collection.find_one(
    {"user_id": user_id},
    sort=[("created_at", -1)]  # Sort by created_at descending
)
        if not video_doc or "evaluation" not in video_doc:
            logger.warning(f"Video evaluation not found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="Video evaluation not found")

        interview_summary = video_doc["evaluation"]
        evaluation_summary = interview_summary.get("summary", " ")
        short_summary = user.get("short_summary", " ").strip()

        logger.debug(f"Fetched short_summary: {short_summary}")
        logger.debug(f"Fetched video evaluation summary: {evaluation_summary}")

        prompt = f"""You are an expert recruiter. Based on the candidates's summary and video evaluation, generate a 7-12 sentence candidate summary in detail.
Start the summary with candidate's name if given. Do not include any formatting or extra explanation.

candidates's summary:
{short_summary}

video Evaluation Summary:
video interview :{evaluation_summary}
"""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are a professional recruiter summarizing candidate profiles."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 250
                }
            ) as response:
                logger.info(f"Azure OpenAI API called for user_id: {user_id}, status: {response.status}")
                
                if response.status != 200:
                    logger.error(f"Azure OpenAI API failed with status {response.status}")
                    raise HTTPException(status_code=500, detail="LLM call failed")

                result = await response.json()
                summary = result["choices"][0]["message"]["content"].strip()
                logger.info(f"Generated summary for user_id {user_id}: {summary}")

                await collection.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$set": {"short_summary": summary}}
                )
                logger.info(f"Updated short_summary in DB for user_id: {user_id}")
                return summary

    except Exception as e:
        logger.error(f"Exception while updating candidate summary for user_id {user_id}: {str(e)}", exc_info=True)
        raise

@app.post("/submit-ticket/")
async def submit_ticket(
    name: str = Form(...),
    email: EmailStr = Form(...),
    phonenumber: str = Form(...),
    description: str = Form(...),
    screenshot: Optional[UploadFile] = File(None)
):
    try:
        collection = db["user-ticket"]
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
    

@app.post("/generate-job-description/")
async def generate_job_description(input_data: Dict = Body(...)):
    """
    Generate a job description based on structured input data.
    """
    try:
        logger.info(f"Generating job description with input: {json.dumps(input_data, indent=2)}")

        # Pass the raw body dict directly
        jd = await call_openai_for_jd(input_data)

        return {
            "status": True,
            "job_description": jd
        }
    except Exception as e:
        logger.error(f"Error generating job description: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate job description.")

@app.post("/update-video-proctoring-logs/")
async def update_video_proctoring_logs(
    user_id: str = Body(...),
    video_url: str = Body(...),
    video_proctoring_logs: Dict = Body(...)
):
    """
    Update video proctoring logs for a user.
    """
    try:
        collection = db["video_proctoring_logs"]
        profile_collection = db["resume_profiles"]
        video_proctoring_logs["user_id"] = user_id
        video_proctoring_logs["updated_at"] = datetime.utcnow()

        await profile_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "video_url": video_url,
                "video_uploaded_at": datetime.now(timezone.utc)
            }}
        )
        # Upsert the log entry
        result = await collection.update_one(
            {"user_id": user_id},
            {"$set": video_proctoring_logs},
            upsert=True
        )

        if result.upserted_id:
            logger.info(f"Inserted new video proctoring log for user {user_id}")
        else:
            logger.info(f"Updated existing video proctoring log for user {user_id}")

        return {"status": True, "message": "Video proctoring logs updated successfully."}
    except Exception as e:
        logger.error(f"Error updating video proctoring logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to update video proctoring logs.")

@app.post("/update-audio-proctoring-logs/")
async def update_audio_proctoring_logs(
    user_id: str = Body(...),
    audio_proctoring_logs: Dict = Body(...)
):
    """
    Update audio proctoring logs for a user.
    """
    try:
        collection = db["audio_proctoring_logs"]
        audio_proctoring_logs["user_id"] = user_id
        audio_proctoring_logs["updated_at"] = datetime.utcnow()

        # Upsert the log entry
        result = await collection.update_one(
            {"user_id": user_id},
            {"$set": audio_proctoring_logs},
            upsert=True
        )

        if result.upserted_id:
            logger.info(f"Inserted new audio proctoring log for user {user_id}")
        else:
            logger.info(f"Updated existing audio proctoring log for user {user_id}")

        return {"status": True, "message": "audio proctoring logs updated successfully."}
    except Exception as e:
        logger.error(f"Error updating audio proctoring logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to update audio proctoring logs.")

@app.get("/contacts-csv/{job_id}")
async def get_contacts_csv(
    job_id: str,
    audio_attended: Optional[bool] = None,
    video_attended: Optional[bool] = None,
    video_interview_sent: Optional[bool] = None,
    application_status: Optional[str] = None,
    shortlisted: Optional[bool] = None,
    call_for_interview: Optional[bool] = None
):
    """
    Get contacts CSV for a specific job role with optional filters.
    Returns a CSV file with Name, Mobile, and Email columns.
    """
    try:
        # Validate job role exists
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job role not found")
        
        # Get all profiles for this job role
        profile_collection = db["resume_profiles"]
        
        # Build filter conditions - EXACT SAME LOGIC as job-candidates endpoint
        filter_conditions = {"job_id": job_id}
        
        if audio_attended is not None:
            if audio_attended:
                filter_conditions["audio_interview"] = True
            else:
                filter_conditions["$or"] = [
                    {"audio_interview": False},
                    {"audio_interview": {"$exists": False}}
                ]
        
        if application_status is not None:
            if application_status == "SendVideoLink":
                filter_conditions["$or"] = [
                    { "video_email_sent": True },
                    { "application_status": "SendVideoLink" }
                ]
            else:
                filter_conditions["application_status"] = application_status
                
        if video_attended is not None:
            if video_attended:
                filter_conditions["video_interview_start"] = True
            else:
                filter_conditions["$or"] = [
                    {"video_interview_start": False},
                    {"video_interview_start": {"$exists": False}}
                ]
                
        if video_interview_sent is not None:
            if video_interview_sent:
                filter_conditions["video_email_sent"] = True
            else:
                filter_conditions["$or"] = [
                    {"video_email_sent": False},
                    {"video_email_sent": {"$exists": False}}
                ]

        if shortlisted is not None:
            if shortlisted:
                filter_conditions["final_shortlist"] = True
            else:
                filter_conditions["$or"] = [
                    {"final_shortlist": False},
                    {"final_shortlist": {"$exists": False}}
                ]
                
        if call_for_interview is not None:
            if call_for_interview:
                filter_conditions["call_for_interview"] = True
            else:
                filter_conditions["$or"] = [
                    {"call_for_interview": False},
                    {"call_for_interview": {"$exists": False}}
                ]
        
        # Log the filter conditions for debugging
        logger.info(f"Filter conditions for contacts CSV: {filter_conditions}")
        
        # If no filters were applied, use the original contacts_csv logic
        if not any([audio_attended, video_attended, video_interview_sent, application_status, shortlisted, call_for_interview]):
            filter_conditions["audio_interview"] = {"$exists": False}
            filter_conditions["$or"] = [
                {"application_status": {"$ne": "Rejected"}},
                {"application_status": {"$exists": False}}
            ]
            logger.info(f"Applied default contacts filter: {filter_conditions}")
        
        # Get profiles with basic information
        cursor = profile_collection.find(
            filter_conditions,
            {
                "_id": 1,
                "basic_information.full_name": 1,
                "basic_information.phone_number": 1,
                "basic_information.email": 1
            }
        )
        
        # Generate CSV content
        csv_output = io.StringIO()
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(["Name", "Mobile", "Email"])
        
        for doc in await cursor.to_list(length=None):
            email = doc.get("basic_information", {}).get("email", None)
            
            if not email:
                continue  # Skip if no email
            
            # Step 2: Check if any other profile exists with same email (only when using default filters)
            if not any([audio_attended, video_attended, video_interview_sent, application_status, shortlisted, call_for_interview]):
                other_profile = await profile_collection.find_one({
                    "basic_information.email": email,
                    "_id": {"$ne": doc["_id"]},  # exclude current profile
                    "$or": [
                        {"audio_interview": {"$exists": True}},
                        {"application_status": "Rejected"}
                    ]
                })
                
                if other_profile:
                    continue  # Skip this profile because of matching rule
            
            name = doc.get("basic_information", {}).get("full_name", "N/A")
            mobile = doc.get("basic_information", {}).get("phone_number", "N/A")
            
            # Clean the phone number
            if mobile != "N/A":
                mobile = re.sub(r'^\+91', '', mobile)  # Remove +91 prefix
                mobile = re.sub(r'[\s,-]', '', mobile)  # Remove spaces, commas, and hyphens
                mobile = re.sub(r'\D', '', mobile)  # Remove any non-digit characters
                if len(mobile) > 10:
                    mobile = mobile[-10:]  # Extract last 10 digits
            
            csv_writer.writerow([name, mobile, email])
        
        csv_content = csv_output.getvalue()
        csv_output.close()
        
        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"contacts_{job_id}_{timestamp}.csv"
        
        # Return CSV as streaming response
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        error_msg = f"Error retrieving contacts CSV: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
@app.post("/reset-video-interview/")
async def reset_video_interview(
    user_id: str = Body(...),
    reset_reason: Optional[str] = Body(None)
):
    """
    Reset the video interview status for a user.
    """
    try:
        collection = db["resume_profiles"]
        
        # Check if user exists
        user_profile = await collection.find_one({"_id": ObjectId(user_id)})
        if not user_profile:
            return JSONResponse(status_code=200, content={"status": False,"message": "User profile not found"}) 
            
        
        # Reset video interview fields
        update_data = {
            "video_interview_start": False,
            "video_url": None,
            "video_uploaded_at": None,
            "video_interview_reset_reason": reset_reason,
            "video_interview_reset_at": datetime.utcnow()
        }
        
        await collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": update_data,
                "$inc": {"video_interview_reset_count": 1}
            }
        )
        
        return {
            "status": True,
            "message": "Video interview reset successfully",
            "user_id": user_id,
            "reset_reason": reset_reason
        }
    except Exception as e:
        error_msg = f"Error resetting video interview: {str(e)}"
        logger.error(error_msg, exc_info=True)

class JobRoleStatus(BaseModel):
    job_id: str
    status: bool 

@app.post("/job-role-status/")
async def update_job_status(request:JobRoleStatus):
    try:
        collection=db["job_roles"]

        job_role = await collection.find_one({"_id": ObjectId(request.job_id)})
        if not job_role:
            return JSONResponse(status_code=200, content={"status": False,"message": "Job role not found"})
        await collection.update_one(
            {"_id": ObjectId(request.job_id)},
            {"$set": {"is_active": request.status, "status_updated_timestamp": datetime.utcnow()}}
        )
        return {
            "status": True,
            "message": "Job role status updated successfully",
            "job_id": request.job_id,
            "new_status": request.status
        }
    except Exception as e:
        error_msg = f"Error updating job role status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class JobFitAssessmentRequest (BaseModel):
    resume_text:str
    jd: str
@app.post("/job_fit_assessment_test/")
async def job_fit_assessment_test(request: JobFitAssessmentRequest):
    try:
        logger.info("Received job fit assessment test request")
        assessment = await generate_job_fit_summary(request.resume_text, request.jd)
        return {
            "status": True,
            "job_fit_assessment": assessment
        }
    except Exception as e:
        error_msg = f"Error in job fit assessment: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


######################################################################
if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
