from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Query, Form, Header, Depends
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Union,Literal
import os
from jose import JWTError, jwt
from dotenv import load_dotenv
from pydantic_models import Item, ResumeData
import aiohttp
import json
from pypdf import PdfReader
import io
import logging
from fastapi.responses import StreamingResponse
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import random
import asyncio
import uuid
import time
from resume_ocr import extract_text_with_ocr
from bson import ObjectId
from fastapi.responses import JSONResponse
from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceExistsError
import base64
from azure.storage.blob import BlobBlock
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from enum import Enum
from azure.storage.blob import ContentSettings
from auth_utils import create_access_token, create_refresh_token ,save_refresh_token, get_current_user, verify_access_token
import bcrypt
import requests
import pytz
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
from sarvamai import SarvamAI
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
    AZURE_STORAGE_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_CONTAINER_NAME", " ")
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
        
        # if len(text) < 500:
        #     logger.warning("Unable to extract sufficient text from PDF")
        #     raise HTTPException(status_code=400, detail="Unable to extract sufficient text from PDF")
        
        # if not is_likely_resume(text):
        #     raise HTTPException(status_code=400, detail="The uploaded document does not appear to be a valid resume")
        
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
    #language: str

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

async def generate_audio_intv_questions(resume_text: str)  -> list[str]:
    prompt = f"""
You are an expert recruiter.
Your task is to generate 3 interview questions tailored to the candidate’s CV.
Follow the rules below carefully.

CV:
{resume_text}

Question Rules

Q1 – Performance / Achievement

If performance metrics, quotas, awards, or notable achievements are present:
"Just went through your CV and saw that you [SPECIFIC_CLAIM] — that’s [positive adjective]! Could you tell me a bit more about that?"


If no performance/achievement data:
“Tell me about an achievement you’re proud of. It would be great if it is something that shows how you approach selling or influencing others."

Q2 – Industry / Buyer Persona

If industry or buyer persona experience is available:
"I see you've been [SPECIFIC_EXPERIENCE] - [ACKNOWLEDGMENT_OF_DIFFICULTY]! What do you typically find is [BUYER_TYPE]'s biggest [CONCERN/CHALLENGE] when [CONTEXT]?"

If no industry/buyer info:
"If you had to sum it up, what do you think is the key to getting someone genuinely interested in what you’re selling or pitching?"

Q3 – Motivation / Future Fit

If company history or industry is available:
"Given your background in [THEIR_CONTEXT], I'm curious - what's drawing you to this opportunity and what do you think would transfer well from your current experience?"

If limited background info:
"What's drawing you to this role and what skills from your background do you think would be most valuable here?"


Only return a valid Python list of string questions.
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
@app.post("/generate-interview-questions/")
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
        
        resume_text = profile.get("resume_text", "")
        questions = await generate_audio_intv_questions(
            resume_text=resume_text
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
    score: int  # rubric-based score (e.g., 0–20 for Q1, 0–15 for Q2, 0–5 for Q3, or 0–100 for others)
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
    language: Optional[str] = "en-IN"  # Language of the user's answers

async def evaluate_answer(question: str, answer: str, i: int) -> AnswerEvaluation:
    logger.info(f"Evaluating answer for question: {question[:100]}...")

    prompt = f"""
Evaluate the following interview Q&A based on the scoring framework for the given question number, and return a JSON response.

Question Number: Q{i}
Question: {question}
Answer: {answer}

### Scoring Frameworks ###

Q1: Performance Validation (20 points)
- STRONG (15–20 points):
  - Provides specific numbers/metrics that match or exceed resume claims
  - Details are consistent and realistic
  - Shows clear ownership of results
  - Natural, confident delivery
- MODERATE (8–14 points):
  - Some specifics but missing depth
  - Numbers roughly align with resume
  - Hesitant but generally consistent
  - Can explain their role in results
- WEAK (0–7 points):
  - Vague, can't provide specific numbers
  - Claims don't match resume
  - Evasive or contradictory responses
  - Takes credit for team/company results inappropriately

Q2: Industry Knowledge (15 points)
- STRONG (12–15 points):
  - Demonstrates deep understanding of buyer concerns
  - Uses correct industry terminology
  - Specific, realistic examples
  - Shows genuine sales experience
- MODERATE (6–11 points):
  - General awareness but limited depth
  - Some industry knowledge
  - Generic but not incorrect responses
  - Shows some real experience
- WEAK (0–5 points):
  - No industry knowledge or incorrect information
  - Generic, rehearsed responses
  - Can't demonstrate actual buyer interaction
  - Clearly fabricated experience

Q3: Intent & Transferability (5 points)
- STRONG (4–5 points):
  - Clear, thoughtful motivation
  - Shows research about the role/company
  - Realistic about skill transfer
  - Growth-oriented mindset
- MODERATE (2–3 points):
  - Reasonable motivation
  - Some understanding of role requirements
  - Generally realistic expectations
  - Generic but acceptable responses
- WEAK (0–1 points):
  - Money-focused only or running from problems
  - No understanding of role/company
  - Unrealistic expectations
  - Red flag motivations

### Required Output (JSON Format) ###
{{
  "score": int,  # Based on the applicable Q1/Q2/Q3 rubric
  "sales_motion": "inbound" | "outbound" | "hybrid" | "not mentioned",
  "sales_cycle": "short" | "medium" | "long" | "not mentioned",
  "icp": "string",  # Mention company sizes, industries, buyer roles
  "highlights": ["string", "string", ...],  # 2–3 clear strengths
  "red_flags": ["string", "string", ...],   # if any
  "coaching_focus": "string",  # Skill to improve (based on response quality)
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
                    "score",
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

                    # Validate numeric score ranges (generic safeguard: 0–100 max, 
                    # but actual max depends on question type — normalization happens later)
                    evaluation["score"] = max(0, min(100, evaluation["score"]))

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

async def audio_updated_fit_assessment (job_fit_assessment: str, audio_interview: str) -> str:
    prompt = f"""
You are an expert recruiter evaluating a candidate’s overall fit by combining their resume assessment and audio interview.

Inputs:
- Resume Assessment (out of 100): {job_fit_assessment}
- Audio Interview Score (out of 40): {audio_interview}
Scoring Framework 
Total Score = Resume Score (100) + Audio Interview Score (40) = 140 points
Final Classifications:
- HIGH FIT: 105–140 points (75%+)
- MEDIUM FIT: 70–104 points (50–74%)
- LOW FIT: 0–69 points (<50%)
Score Adjustment Logic
Upgrade Tags:
- "HIDDEN GEM": Resume = Low/Medium but Audio Interview = Strong → Upgrade 1 level
Downgrade Tags:
- "RESUME PADDER": Any resume score + Weak Audio → Downgrade to Low
Confirmation Tags:
- "VERIFIED": High/Medium Resume + Good Audio → Fast track confirmation
Rationale (3–5 bullets):
- Provide a brief rationale for the final classification
-justify the change if upgraded/downgraded
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
        job_fit_assessment = profile.get("job_fit_assessment", "") if profile else ""

        if not profile:
            return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)

        results = []
        scores = []  # store rubric-based scores (Q1–Q3 or others if extended)
        strengths = []
        areas_for_improvement = []
        red_flags = []
        sales_motions = []
        sales_cycles = []
        icp_entries = []
        coaching_focus = []

        for idx, qa_pair in enumerate(evaluation.qa_pairs):
            i = idx + 1  # Question number (1-based)
            evaluation_result = await evaluate_answer(qa_pair.question, qa_pair.answer, i)

            results.append({
                "question": qa_pair.question,
                "answer": qa_pair.answer,
                "evaluation": evaluation_result.dict()
            })
            logger.info(f"Evaluated Q{i}")

            # Collect score
            if hasattr(evaluation_result, "score"):
                scores.append(evaluation_result.score)

            # Collect supporting data
            if evaluation_result.score >= (0.75 * (20 if i == 1 else 15 if i == 2 else 5 if i == 3 else 100)):
                strengths.append(evaluation_result.fit_summary)

            if evaluation_result.score <= (0.4 * (20 if i == 1 else 15 if i == 2 else 5 if i == 3 else 100)):
                areas_for_improvement.append(evaluation_result.fit_summary)

            if evaluation_result.red_flags:
                red_flags.extend(evaluation_result.red_flags)

            if evaluation_result.sales_motion != "not mentioned":
                sales_motions.append(evaluation_result.sales_motion)

            if evaluation_result.sales_cycle != "not mentioned":
                sales_cycles.append(evaluation_result.sales_cycle)

            if evaluation_result.icp:
                icp_entries.append(evaluation_result.icp)

            if evaluation_result.coaching_focus:
                coaching_focus.append(evaluation_result.coaching_focus)

        # Compute averages
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0

        # Define threshold based on rubric max (Q1=20, Q2=15, Q3=5, others assumed /100)
        # For simplicity, normalize all scores to percentage scale
        normalized_scores = []
        for idx, score in enumerate(scores):
            q_num = idx + 1
            max_score = 20 if q_num == 1 else 15 if q_num == 2 else 5 if q_num == 3 else 100
            normalized_scores.append((score / max_score) * 100)

        avg_normalized = round(sum(normalized_scores) / len(normalized_scores), 2) if normalized_scores else 0

        audio_interview_status = avg_normalized >= 65  # keep threshold consistent

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
        "average_score": avg_score,  # raw average (rubric-based, mixed scales)
        "average_normalized_score": avg_normalized,  # normalized to 0–100
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
        audio_updated_job_fit_assessment=await audio_updated_fit_assessment(job_fit_assessment=job_fit_assessment, audio_interview=str(evaluation_doc))
        result = await collection.update_one(
            {"_id": ObjectId(evaluation.user_id)},
            {"$set": {"audio_updated_job_fit_assessment": audio_updated_job_fit_assessment}}
        )
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
    total_question_score = 0

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
            logger.info(f"Evaluating Q{q_num} with type {evaluation_type}")
            logger.info(f"Evaluation for Q{q_num}: {evaluation}")   
            question_evaluation = {
                "question_number": q_num,
                "question": question_text,
                "answer": user_answer,
                "skill_score": evaluation.get("skill_score", 0),
                "trait_score": evaluation.get("trait_score", 0),
                "skill_reasoning": evaluation.get("skill_reasoning", ""),
                "trait_reasoning": evaluation.get("trait_reasoning", ""),
                "has_signal": evaluation.get("has_signal", True),
                "question_score": evaluation.get("question_score", 0),
                "timestamp": answer.get("timestamp", datetime.utcnow())
            }
            evaluated_questions.append(question_evaluation)

            if evaluation.get("has_signal", True):
                total_skill_score += evaluation.get("skill_score", 0)
                total_trait_score += evaluation.get("trait_score", 0)
                total_question_score += evaluation.get("question_score", 0)
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
        evaluated_questions, avg_skill_score, avg_trait_score, resume_text, role, job_config,total_question_score
    )
    
    evaluation_result = {
        "session_id": session_id,
        "question_evaluations": evaluated_questions,
        "overall_scores": {
            "average_skill_score": round(avg_skill_score, 2),
            "average_trait_score": round(avg_trait_score, 2),
            "total_questions": len(interview_questions)+1,
            "total_question_score": round(total_question_score, 0),
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
        "question_score": [0-5 based on rubric scoring_logic]
        "skill_reasoning": "Brief explanation of skill score based on the rubric's competencies and the answer.",
        "trait_reasoning": "Brief explanation of trait score based on the rubric's characteristics and the answer.",
        "has_signal": [true/false],
        
    }}

    Focus on practical ability, real-world experience, and the mindset needed for success in this role.
    When determining scores, prioritize adherence to the structured 'scoring_logic' within the rubric.
    """
    logger.info(f"Evaluating Question {question_number} with rubric: {rubric_str}")
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
                        "question_score": evaluation.get("question_score", 0),
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
            "question_score": 0,
            "skill_reasoning": f"Evaluation error: {e}",
            "trait_reasoning": f"Evaluation error: {e}",
            "has_signal": False
        }


async def generate_interview_summary(evaluated_questions: list, avg_skill_score: float, avg_trait_score: float, resume_text: str, role: str, job_config: dict,total_question_score: float):
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

    #total_overall_score = total_skill_score + total_trait_score_sum
    total_overall_score=total_question_score
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
    - Total Interview Score: {total_overall_score}

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

    **Reasoning:** [1-2 sentences explaining the recommendation based on Total Interview Score and specific observations from the interview.]
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

        first_question = interview_questions[0]
        await collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"last_question": first_question["question"]}}
        )
        await resume_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "video_interview_start": True,
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
            key_highlights= await generate_key_highlights(session_id)
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

async def generate_key_highlights(session_id: str):
    """
    Extract key highlights from interview answers stored in the database.
    """
    logger.info(f"Extracting interview highlights for session {session_id}")
    
    try:
        # Fetch session from database
        collection = db["interview_sessions"]
        session = await collection.find_one({"_id": ObjectId(session_id)})
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        user_id = session.get("user_id", None)

        # Extract answers
        answers = session.get("answers", [])
        if not answers:
            raise HTTPException(status_code=400, detail="No answers found in session")
        
        # Prepare answers text
        answers_text = "\n".join([
            f"Question: {answer.get('question', '')}\nAnswer: {answer.get('answer', '')}"
            for answer in answers if answer.get('type') != 'final_thoughts'
        ])
        
        # Prompt only for highlights
        prompt = f"""
        Based on the following interview answers, extract the key highlights of the candidate's performance.
        Summarize strengths, unique points, or standout elements in bullet points.

        Interview Answers:
        {answers_text}

        Return only bullet points in plain text, no JSON.
        Example:
        - Strong ability to explain complex ideas clearly
        - Shows confidence and authenticity in responses
        - Demonstrates good audience awareness
        """
        
        # Call Azure OpenAI
        async with aiohttp.ClientSession() as client:
            async with client.post(
                AZURE_OPENAI_URL,
                headers=AZURE_HEADERS,
                json={
                    "messages": [
                        {"role": "system", "content": "You are an expert interviewer extracting highlights from answers."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 600
                }
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Failed to extract highlights")
                
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Save highlights to DB
                await collection.update_one(
                    {"_id": ObjectId(session_id)},
                    {"$set": {"interview_highlights": content}}
                )
                
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "true",
                        "message": "highlights extracted successfully",
                        "user_id": user_id,
                        "highlights": content
                    }
                )
                
    except Exception as e:
        error_msg = f"Error extracting highlights: {str(e)}"
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
    work_preference: str
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
Provide a Job Fit Assessment for this resume and job description using the following weighted scoring system:
Resume:
{resume_text}
Job Description:
{job_description}
Scoring Criteria (Total 100 points):
EXPERIENCE MATCH (15 points)
- Sales Years:
  - Meets requirement exactly = 15 pts
  - Exceeds by 1-2 years = 15 pts
  - Exceeds by 3+ years = 10 pts (overqualified)
  - 1 year short = 5 pts
  - 2+ years short = 0 pts
MARKET ALIGNMENT (40 points)
- Industry Match:
  - Exact match = 15 pts
  - Related industry = 10 pts
  - Different but transferable = 5 pts
  - Completely different = 0 pts
- Product Type:
  - Same product category = 15 pts
  - Similar complexity = 10 pts
  - Different product = 3 pts
  - No product sales experience = 0 pts
- Buyer Persona:
  - Same decision makers = 10 pts
  - Similar seniority = 6 pts
  - Different buyers = 2 pts
  - No B2B experience = 0 pts
PERFORMANCE MATCH (25 points)
- Quota Achievement:
  - Consistently >110% = 25 pts
  - 100–110% average = 20 pts
  - 90–100% = 10 pts
  - <90% or no data = 0 pts
REQUIREMENTS FIT (20 points)
- Tool Proficiency:
  - Has required tools = 8 pts
  - Has similar tools = 4 pts
  - No relevant tools = 0 pts
- Sales Process:
  - Matches exactly = 7 pts
  - Transferable skills = 4 pts
  - Different process = 1 pt
- Compensation:
  - Within budget = 5 pts
  - 10–20% over = 2 pts
  - 20%+ over = 0 pts
Classification:
- HIGH FIT: 70–100 points
- MEDIUM FIT: 40–69 points
- LOW FIT: 0–39 points
Output Format:
Job Fit Assessment: [HIGH/MEDIUM/LOW]
Total Score: [X/100]
Score Breakdown:
- Experience Match: [X/15]
- Market Alignment: [X/40]
- Performance Match: [X/25]
- Requirements Fit: [X/20]
Rationale (3–5 bullets):
- Compare experience years vs. requirement
- Industry/product/buyer persona alignment
- Sales skills & quota achievement evidence
- Tools/process/compensation alignment
- Highlight promotions or multiple roles at the same company (if applicable)
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
async def upload_to_blob_storage(file: UploadFile, user_id: str) -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    blob_name = f"interview-video-{user_id}-{timestamp}{file_extension}"

    blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(settings.AZURE_STORAGE_CONTAINER_NAME)

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
        
        # Upload to blob storage
        video_url, blob_name = await upload_to_blob_storage(file, user_id)
        
        # Store video information in database
        video_doc = {
            "user_id": user_id,
            "video_url": video_url,
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

from fastapi import Query, HTTPException
from bson import ObjectId
from datetime import datetime

@app.get("/all-jobs")
async def get_all_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    search: Optional[str] = Query(None, description="Search term for job title or company name")
):
    """
    Get all active jobs (new format only) with hiring manager details.
    Does not use company_profiles. Uses hiring_managers collection instead.
    """
    logger.info(f"Fetching all NEW FORMAT jobs (page {page}, size {page_size})")

    try:
        job_collection = db["job_roles"]
        hiring_manager_collection = db["hiring_managers"]

        # Filter only new-format jobs
        filter_conditions = {
            "is_active": True,
            "basicInfo": {"$exists": True}
        }

        # Add search filter
        if search:
            filter_conditions["$or"] = [
                {"basicInfo.jobTitle": {"$regex": search, "$options": "i"}},
                {"basicInfo.companyName": {"$regex": search, "$options": "i"}},
                {"experienceSkills.skillsRequired": {"$regex": search, "$options": "i"}}
            ]

        # Pagination setup
        total_jobs = await job_collection.count_documents(filter_conditions)
        total_pages = (total_jobs + page_size - 1) // page_size
        if page > total_pages and total_pages > 0:
            raise HTTPException(status_code=400, detail=f"Page number exceeds total pages ({total_pages})")
        skip = (page - 1) * page_size

        # Fetch jobs
        jobs = await job_collection.find(filter_conditions).skip(skip).limit(page_size).to_list(length=page_size)
        logger.info(f"Found {len(jobs)} jobs after pagination")

        jobs_with_manager = []

        for job in jobs:
            basic_info = job.get("basicInfo", {})
            exp_skills = job.get("experienceSkills", {})
            compensations = job.get("compensations", {})

            # Identify hiring manager from created_by
            manager_id = job.get("created_by")
            manager = None
            if manager_id:
                try:
                    manager = await hiring_manager_collection.find_one({"_id": ObjectId(manager_id)})
                except Exception:
                    manager = None

            # Build final job response
            basic_info = job.get("basicInfo") or {}
            exp_skills = job.get("experienceSkills") or {}
            compensations = job.get("compensations") or {}

            job_with_manager = {
                "job_id": str(job.get("_id", "")),
                "job_title": basic_info.get("jobTitle", ""),
                "company_name": basic_info.get("companyName", ""),
                "role_type": basic_info.get("roleType", ""),
                "primary_focus": basic_info.get("primaryFocus", []),
                "sales_process_stages": basic_info.get("salesProcessStages", []),
                "min_experience": exp_skills.get("minExp", ""),
                "max_experience": exp_skills.get("maxExp", ""),
                "skills_required": exp_skills.get("skillsRequired", []),
                "work_location": exp_skills.get("workLocation", ""),
                "locations": exp_skills.get("location", []),
                "time_zones": exp_skills.get("timeZone", []),
                "base_salary": compensations.get("baseSalary", {}),
                "ote": compensations.get("ote", []),
                "opportunities": compensations.get("opportunities", []),
                "languages": compensations.get("laguages", []),
                "created_at": (
                    job.get("created_at").isoformat()
                    if isinstance(job.get("created_at"), datetime)
                    else ""
                ),
                "hiring_manager": {
                    "manager_id": str(manager["_id"]) if manager else str(manager_id) if manager_id else "",
                    "first_name": manager.get("first_name", "") if manager else "",
                    "last_name": manager.get("last_name", "") if manager else ""
                }
            }

            jobs_with_manager.append(job_with_manager)

        # Pagination metadata
        response = {
            "status": True,
            "message": "Jobs retrieved successfully",
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_jobs": total_jobs,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "jobs": jobs_with_manager
        }

        return response

    except Exception as e:
        logger.error(f"Error retrieving jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving jobs: {str(e)}")
@app.get("/job-candidates/{job_id}")
async def get_job_candidates(
    job_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    authorization: str = Header(...),
    page_size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    audio_attended: Optional[bool] = Query(None, description="Filter by audio interview status"),
    seen: bool | None = None,
    video_attended: Optional[bool] = Query(None, description="Filter by video upload status"),
    video_interview_sent: Optional[bool] = Query(None, description="Filter by video interview sent status"),
    application_status: Optional[str] = Query(None, description="Filter by application status"),
    shortlisted: Optional[bool] = Query(None, description="Filter by shortlisted status"),
    call_for_interview: Optional[bool] = Query(None, description="Filter by call for interview status"),
    searchQuery: Optional[str] = Query(None, description="Search across name, phone, email, LinkedIn"),
    location: Optional[str] = Query(None, description="Filter by current location")
):
    """
    Get candidates for a specific job role with optional filters.
    """
    try:
        # --- Verify access token ---
        if not authorization.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"status": False, "message": "Invalid authorization header"})
        token = authorization.split(" ")[1]

        try:
            payload = verify_access_token(token)
            admin_id = payload.get("sub")
            role = payload.get("role")
            if not admin_id or role != "superadmin":
                return JSONResponse(status_code=403, content={"status": False, "message": "Unauthorized"})
        except Exception:
            return JSONResponse(status_code=401, content={"status": False, "message": "Invalid or expired token"})
        # Validate job role exists
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job role not found")
        # if not job.get("is_active", True):
        #     raise HTTPException(status_code=400, detail="Inactive job role")
        
        # Get all profiles for this job role
        # --- Build filters ---
        # --- Build filters ---
        # --- Build filters ---
        profile_collection = db["resume_profiles"]
        user_collection = db["user_accounts"]

        # Initialize profile filter conditions
        profile_and_conditions = []
        profile_and_conditions.append({"job_id": job_id})

        # Add profile-level filters
        if audio_attended is not None:
            profile_and_conditions.append({"audio_interview": True} if audio_attended else {"audio_interview": {"$ne": True}})

        if seen is not None:
            profile_and_conditions.append({"seen_by_manager": True} if seen else {"seen_by_manager": {"$ne": True}})

        if application_status is not None:
            if application_status == "SendVideoLink":
                profile_and_conditions.append({
                    "$or": [
                        {"video_email_sent": True},
                        {"application_status": "SendVideoLink"}
                    ]
                })
            else:
                profile_and_conditions.append({"application_status": application_status})

        if video_attended is not None:
            profile_and_conditions.append({"video_interview_start": True} if video_attended else {"video_interview_start": {"$ne": True}})

        if video_interview_sent is not None:
            profile_and_conditions.append({"video_email_sent": True} if video_interview_sent else {"video_email_sent": {"$ne": True}})

        if shortlisted is not None:
            profile_and_conditions.append({"final_shortlist": True} if shortlisted else {"final_shortlist": {"$ne": True}})

        if call_for_interview is not None:
            profile_and_conditions.append({"call_for_interview": True} if call_for_interview else {"call_for_interview": {"$ne": True}})

        # --- Handle searchQuery and location (user_accounts fields) ---
        matching_user_ids = None

        if searchQuery or location:
            user_filter_conditions = []
            
            # Search query across name, email, phone, linkedin
            if searchQuery:
                regex = re.compile(re.escape(searchQuery), re.IGNORECASE)
                user_filter_conditions.append({
                    "$or": [
                        {"name": regex},
                        {"email": regex},
                        {"phone": regex},
                        {"basic_information.full_name": regex},
                        {"linkedin_profile": regex}
                    ]
                })
            
            # Location filter
            if location:
                loc_regex = re.compile(re.escape(location), re.IGNORECASE)
                user_filter_conditions.append({"basic_information.current_location": loc_regex})
            
            # Build user filter
            if len(user_filter_conditions) == 1:
                user_filter = user_filter_conditions[0]
            else:
                user_filter = {"$and": user_filter_conditions}
            
            # Get matching user IDs
            matching_users = await user_collection.find(user_filter, {"_id": 1}).to_list(None)
            matching_user_ids = [str(u["_id"]) for u in matching_users]
            
            # If no users match, return empty result early
            if not matching_user_ids:
                return {
                    "status": True,
                    "message": "No candidates found",
                    "job_details": {"title": job.get("basicInfo", {}).get("jobTitle", "")},
                    "pagination": {"total_candidates": 0, "total_pages": 0},
                    "candidates": []
                }
            
            # Add user_id filter to profile conditions
            profile_and_conditions.append({"user_id": {"$in": matching_user_ids}})

        # Combine profile conditions
        if len(profile_and_conditions) == 1:
            filter_conditions = profile_and_conditions[0]
        else:
            filter_conditions = {"$and": profile_and_conditions}

        # --- Parallel count queries ---
        base_filter = {"$and": profile_and_conditions} if len(profile_and_conditions) > 1 else profile_and_conditions[0]

        audio_count_conditions = profile_and_conditions.copy()
        audio_count_conditions.append({"audio_interview": True})

        video_count_conditions = profile_and_conditions.copy()
        video_count_conditions.append({"video_interview_start": True})

        video_round_conditions = profile_and_conditions.copy()
        video_round_conditions.append({
            "$or": [{"video_email_sent": True}, {"application_status": "SendVideoLink"}]
        })

        count_tasks = [
            profile_collection.count_documents(base_filter),
            profile_collection.count_documents({"$and": audio_count_conditions}),
            profile_collection.count_documents({"$and": video_count_conditions}),
            profile_collection.count_documents({"$and": video_round_conditions})
        ]

        total_candidates, audio_attended_count, video_attended_count, moved_to_video_round_count = await asyncio.gather(*count_tasks)

# ... rest of your code remains the same ...

        total_pages = (total_candidates + page_size - 1) // page_size
        if page > total_pages and total_pages > 0:
            raise HTTPException(status_code=400, detail=f"Page number exceeds total pages ({total_pages})")

        skip = (page - 1) * page_size

        # --- Fetch profiles in bulk ---
        profiles = await profile_collection.find(
            filter_conditions,
            {"user_id": 1, "application_status": 1, "final_shortlist": 1, "call_for_interview": 1,"seen_by_manager": 1,"job_fit_assessment":1, "audio_updated_job_fit_assessment":1,
             "audio_interview": 1, "audio_url": 1, "video_url": 1, "video_email_sent": 1, "created_at": 1,"processed_video_url": 1,
             "career_overview": 1}
        ).sort("created_at", -1).skip(skip).limit(page_size).to_list(length=page_size)

        if not profiles:
            return {
                "status": True,
                "message": "No candidates found",
                "job_details": {"title": job.get("basicInfo", {}).get("jobTitle", "")},
                "pagination": {"total_candidates": 0, "total_pages": 0},
                "candidates": []
            }

        user_ids = [ObjectId(p["user_id"]) for p in profiles if p.get("user_id")]
        profile_ids = [str(p["_id"]) for p in profiles]

        # --- Prefetch all dependent data concurrently ---
        user_collection = db["user_accounts"]
        interview_collection = db["interview_sessions"]
        sales_collection = db["sales_scenarios"]
        audio_collection = db["audio_interview_results"]
        audio_proctoring_collection = db["audio_proctoring_logs"]
        video_proctoring_collection = db["video_proctoring_logs"]

        user_task = user_collection.find({"_id": {"$in": user_ids}}).to_list(None)
        interview_task = interview_collection.find({"application_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None)
        audio_task = audio_collection.find({"application_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None)
        audio_proc_task = audio_proctoring_collection.find({"user_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None)
        video_proc_task = video_proctoring_collection.find({"user_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None)
        sales_task = sales_collection.find({"user_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None)

        (
            users,
            interviews,
            audio_interviews,
            audio_proc,
            video_proc,
            sales_scenarios
        ) = await asyncio.gather(user_task, interview_task, audio_task, audio_proc_task, video_proc_task, sales_task)

        # --- Convert to lookup dicts for O(1) access ---
        user_map = {str(u["_id"]): u for u in users}
        interview_map = {i["application_id"]: i for i in interviews}
        audio_map = {a["application_id"]: a for a in audio_interviews}
        audio_proc_map = {p["user_id"]: p for p in audio_proc}
        video_proc_map = {p["user_id"]: p for p in video_proc}
        sales_map = {s["user_id"]: s for s in sales_scenarios}

        candidates = []
        now = datetime.utcnow()

        for profile in profiles:
            user = user_map.get(str(profile.get("user_id")))
            if not user:
                continue

            # --- Compute experience ---
            career_overview = profile.get("career_overview", {})
            company_history = career_overview.get("company_history", [])
            if company_history:
                company_history.sort(key=lambda x: x.get("start_date", ""), reverse=True)
            total_months = 0
            for role in company_history:
                try:
                    start = datetime.strptime(role.get("start_date", ""), "%Y-%m-%d")
                    if role.get("is_current", False):
                        total_months += (now.year - start.year) * 12 + (now.month - start.month)
                    elif role.get("end_date"):
                        end = datetime.strptime(role["end_date"], "%Y-%m-%d")
                        total_months += (end.year - start.year) * 12 + (end.month - start.month)
                except Exception:
                    continue
            career_overview["total_years_experience"] = round(total_months / 12, 1)

            # --- Assemble candidate ---
            candidate = {
                "application_id": str(profile["_id"]),
                "profile_created_at": profile.get("created_at").isoformat() if profile.get("created_at") else None,
                "user_id": str(user["_id"]),
                "name": user.get("name", ""),
                "email": user.get("email", ""),
                "phone": user.get("phone", ""),
                "linkedin_url": user.get("linkedin_profile", ""),
                "professional_summary": user.get("professional_summary", ""),
                "basic_information": user.get("basic_information", {}),
                "career_overview": user.get("career_overview", {}),
                "role_process_exposure": user.get("role_process_exposure", {}),
                "sales_context": user.get("sales_context", {}),
                "tools_platforms": user.get("tools_platforms", {}),
                "resume_url": user.get("resume_url", None),
                "application_status": profile.get("application_status", ""),
                "job_fit_assessment": profile.get("job_fit_assessment", ""),
                "audio_updated_job_fit_assessment": profile.get("audio_updated_job_fit_assessment", ""),
                "final_shortlist": profile.get("final_shortlist", False),
                "call_for_interview": profile.get("call_for_interview", False),
                "seen_by_manager": profile.get("seen_by_manager", False),
                "interview_status": {
                    "audio_interview_passed": profile.get("audio_interview", False),
                    "video_interview_attended": bool(profile.get("processed_video_url")),
                    "audio_interview_attended": bool(profile.get("audio_url")),
                    "video_email_sent": profile.get("video_email_sent", False),
                    "video_interview_url": profile.get("video_url"),
                    "audio_interview_url": profile.get("audio_url"),
                    "processed_video_url": profile.get("processed_video_url"),
                    "resume_url_from_user_account": user.get("resume_url")
                }
            }

            if interview_map.get(str(profile["_id"])):
                i = interview_map[str(profile["_id"])]
                candidate["interview_details"] = {
                    "session_id": str(i["_id"]),
                    "created_at": i["created_at"].isoformat(),
                    "communication_evaluation": i.get("communication_evaluation", {}),
                    "key_highlights": i.get("interview_highlights", ""),
                    "qa_evaluations": i.get("evaluation", [])
                }

            if audio_map.get(str(profile["_id"])):
                a = audio_map[str(profile["_id"])]
                candidate["audio_interview_details"] = {
                    "audio_interview_id": str(a["_id"]),
                    "created_at": a["created_at"].isoformat(),
                    "qa_evaluations": a.get("qa_evaluations", {}),
                    "audio_interview_summary": a.get("interview_summary", [])
                }

            if audio_proc_map.get(str(profile["_id"])):
                candidate["audio_proctoring_details"] = serialize_document(dict(audio_proc_map[str(profile["_id"])]))

            if video_proc_map.get(str(profile["_id"])):
                candidate["video_proctoring_details"] = serialize_document(dict(video_proc_map[str(profile["_id"])]))

            if sales_map.get(str(profile["_id"])):
                s = sales_map[str(profile["_id"])]
                candidate["sales_scenario_details"] = {
                    "session_id": str(s["_id"]),
                    "created_at": s["created_at"].isoformat(),
                    "sales_conversation_evaluation": s.get("sales_conversation_evaluation", {}),
                    "responses": s.get("responses", [])
                }

            candidates.append(candidate)

        return {
            "status": True,
            "message": "Candidates retrieved successfully",
            "job_details": {
                "title": job["basicInfo"]["jobTitle"],
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
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "candidates": candidates
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error retrieving job candidates: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": False, "message": str(e)})


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
            "status": True,
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
        if matching_profile.get("video_interview_start", True):
            return VideoInterviewLoginResponse(
                status=False,
                message="You have alredy atempted this interview",
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
    Get detailed information about a specific NEW FORMAT job including hiring manager details.
    Does not use company_profiles.
    """
    try:
        job_collection = db["job_roles"]
        hiring_manager_collection = db["hiring_managers"]

        # Find the job (only active, new format)
        job = await job_collection.find_one({
            "_id": ObjectId(job_id),
            "is_active": True,
            "basicInfo": {"$exists": True}
        })

        if not job:
            return JSONResponse(status_code=404, content={"status": False, "message": "Job not found"})

        # Identify hiring manager
        manager_id = job.get("created_by")
        manager = None
        if manager_id:
            try:
                manager = await hiring_manager_collection.find_one({"_id": ObjectId(manager_id)})
            except Exception:
                manager = None

        # Extract sections safely
        basic_info = job.get("basicInfo", {})
        exp_skills = job.get("experienceSkills", {})
        compensations = job.get("compensations", {})

        # Build response
        job_details = {
            "status": True,
            "message": "Job details retrieved successfully",
            "job": {
                "job_id": str(job.get("_id", "")),
                "job_title": basic_info.get("jobTitle", ""),
                "company_name": basic_info.get("companyName", ""),
                "role_type": basic_info.get("roleType", ""),
                "primary_focus": basic_info.get("primaryFocus", []),
                "sales_process_stages": basic_info.get("salesProcessStages", []),
                "min_experience": exp_skills.get("minExp", ""),
                "max_experience": exp_skills.get("maxExp", ""),
                "skills_required": exp_skills.get("skillsRequired", []),
                "work_location": exp_skills.get("workLocation", ""),
                "locations": exp_skills.get("location", []),
                "time_zones": exp_skills.get("timeZone", []),
                "base_salary": compensations.get("baseSalary", {}),
                "ote": compensations.get("ote", []),
                "opportunities": compensations.get("opportunities", []),
                "key_challenges":compensations.get("keyChallenged", []),
                "languages": compensations.get("laguages", []),
                "created_at": (
                    job.get("created_at").isoformat()
                    if isinstance(job.get("created_at"), datetime)
                    else ""
                ),
                "hiring_manager": {
                    "manager_id": str(manager["_id"]) if manager else str(manager_id) if manager_id else "",
                    "first_name": manager.get("first_name", "") if manager else "",
                    "last_name": manager.get("last_name", "") if manager else "",

                }
            }
        }

        return JSONResponse(status_code=200, content=job_details)

    except Exception as e:
        logger.error(f"Error retrieving job details: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving job details: {str(e)}")

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
            "status": True,
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


search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME", "scooty-search-service")
admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "your-admin-key")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "jobs-index")

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
                return {
                    "status": True,
                    "answer": result["choices"][0]["message"]["content"].strip()
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

sarvam_client = SarvamAI(
    api_subscription_key="sk_qq3znrij_YTENtcZNI9Y5mRONLemRHIru",
)
def translate_questions(input: str, language: str) -> str:
    """
    Translate a single question string to the target language.
    """
    try:
        if not isinstance(input, str):
            raise ValueError(f"translate_questions expected a string, got {type(input)}")

        client = SarvamAI(
            api_subscription_key="sk_qq3znrij_YTENtcZNI9Y5mRONLemRHIru",
        )

        output = client.text.translate(
            input=input,
    source_language_code="en-IN",
    target_language_code=language,
    model="mayura:v1",
    mode="modern-colloquial"
        )

        if not hasattr(output, "translated_text"):
            raise ValueError("SarvamAI did not return translated_text")

        return output.translated_text

    except Exception as e:
        logger.error(f"Error translating text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error translating text")

def translate_to_english(input: str, source_language: str) -> str:
    """
    Translate text from source language to English.
    """
    try:
        if not isinstance(input, str):
            raise ValueError(f"translate_to_english expected a string, got {type(input)}")

        client = SarvamAI(
            api_subscription_key="sk_qq3znrij_YTENtcZNI9Y5mRONLemRHIru",
        )

        output = client.text.translate(
            input=input,
            source_language_code=source_language,
            target_language_code="en-IN",
            model="mayura:v1",
            mode="modern-colloquial"
        )

        if not hasattr(output, "translated_text"):
            raise ValueError("SarvamAI did not return translated_text")

        return output.translated_text

    except Exception as e:
        logger.error(f"Error translating text to English: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error translating text to English")

def translate_to_english(input: str, language: str) -> str:
    """
    Translate text from given language to English.
    """
    try:
        if not isinstance(input, str):
            raise ValueError(f"translate_to_english expected a string, got {type(input)}")

        client = SarvamAI(
            api_subscription_key="sk_qq3znrij_YTENtcZNI9Y5mRONLemRHIru",
        )

        output = client.text.translate(
            input=input,
            source_language_code=language,
            target_language_code="en-IN",
            model="mayura:v1",
            mode="modern-colloquial"
        )

        if not hasattr(output, "translated_text"):
            raise ValueError("SarvamAI did not return translated_text")

        return output.translated_text

    except Exception as e:
        logger.error(f"Error translating text to English: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error translating text to English")

def transliterate_questions(input: str, language: str) -> str:
    """
    Transliterate a single question string to Latin script.
    """
    try:
        if not isinstance(input, str):
            raise ValueError(f"transliterate_questions expected a string, got {type(input)}")

        client = SarvamAI(
            api_subscription_key="sk_qq3znrij_YTENtcZNI9Y5mRONLemRHIru",
        )

        response = client.text.transliterate(
    input=input,
    source_language_code=language,
    target_language_code="en-IN",
    spoken_form=True
)
        return response.transliterated_text
    except Exception as e:
        logger.error(f"Error transliterating text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error transliterating text")

@app.post("/gggenerate-interview-questions/")
async def gen_questions(request: InterviewQuestionRequest):
    """
    Generate interview questions and translate them if needed.
    """
    try:
        collection = db["resume_profiles"]
        profile = await collection.find_one({"_id": ObjectId(request.profile_id)})

        if not profile:
            return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)

        if "interview_otp" in profile:
            return JSONResponse(content={"message": "Audio round already completed", "status": False}, status_code=200)

        # Generate the questions (list of strings)
        questions = await gen_interview_questions(request.profile_id, request.posting_title)

        # Translate each question (if needed)
        translated_questions = [
            translate_questions(q, request.language) for q in questions
        ]
        translitrated_questions = [
             transliterate_questions(q, request.language) for q in translated_questions
        ]
        return {"questions":  translitrated_questions}
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating questions")

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


async def generate_video_highlights(user_id: str) -> str:
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
        raise HTTPException(status_code=500, detail=" ")


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
        await notify_developer_of_new_ticket(ticket_dict)
        await send_support_conformation_email(email, reference_number, name)
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
            "processed_video_url": None,
            "video_interview_reset_reason": reset_reason,
            "video_interview_reset_at": datetime.utcnow()
        }
        
        await collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data,
             "$inc": {"video_interview_reset_count": 1}}
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

from audio_intv_a import start_audio_interview_call
class AudioInterviewCallRequest(BaseModel):
    profile_id: str
@app.post("/start-audio-call/")
async def start_audio_call(request: AudioInterviewCallRequest):
    """
    Start an audio interview call with the candidate.
    """
    try:
        collection = db["resume_profiles"]
        profile = await collection.find_one({"_id": ObjectId(request.profile_id)})

        if not profile:
            return JSONResponse(content={"message": "Resume profile not found"}, status_code=200)

        if "interview_otp" in profile:
            return JSONResponse(content={"message": "Audio round already completed", "status": False}, status_code=200)

        candidate_name = profile.get("name", "Candidate")
        phone_number = profile.get("phone")
        resume_text = profile.get("resume_text", "")

        if not phone_number:
            return JSONResponse(content={"message": "Candidate phone number not available"}, status_code=200)

        call_response = start_audio_interview_call(phone_number, resume_text, candidate_name)
        logger.info(f"Audio interview call response: {call_response}")

        if call_response.get("status") and "call_id" in call_response:
            otp = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            await collection.update_one(
                {"_id": ObjectId(request.profile_id)},
                {
                    "$set": {
                       "interview_otp": otp,
                        "audio_call_id": call_response["call_id"],
                        "audio_call_started_at": datetime.utcnow(),
                        "audio_interview": True,
                        "video_interview_start": False
                    }
                }
            )
            return {
                "status": True,
                "message": "Audio interview call started successfully",
                "call_details": call_response
            }
        else:
            return JSONResponse(content={"message": "Failed to start audio interview call"}, status_code=500)

    except Exception as e:
        logger.error(f"Error starting audio interview call: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error starting audio interview call")

calls_collection = db["resume_profiles"]  # Your collection where call_id is stored

# VAPI configuration
VAPI_TOKEN = "7b38bd97-6291-453e-91f5-0301f82efd4c"
@app.get("/api/user/{user_id}/call-details")
async def get_call_details_by_user_id(user_id: str):
    """
    Fetch call transcript and recording by user _id
    Steps:
    1. Find call_id from database using user_id
    2. Fetch call details from VAPI using call_id
    3. Return transcript, recording URL, and call metadata
    """
    try:
        # Convert string to ObjectId for MongoDB query
        try:
            obj_id = ObjectId(user_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid user ID format")
        
        # Find the document with this user_id
        document = await calls_collection.find_one({"_id": obj_id})
        
        if not document:
            raise HTTPException(status_code=404, detail="User or call record not found")
        
        call_id = document.get("audio_call_id")
        if not call_id:
            raise HTTPException(status_code=404, detail="Call ID not found for this user")
        
        # Fetch call details from VAPI
        headers = {
            "Authorization": f"Bearer {VAPI_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"https://api.vapi.ai/call/{call_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Failed to fetch call from VAPI: {response.text}"
            )
        
        call_data = response.json()
        
        # Extract key information
        result = {
            "user_id": user_id,
            "call_id": call_id,
            "candidate_name": document.get("candidate_name"),
            "phone_number": document.get("phone_number"),
            "transcript": call_data.get("transcript"),
            "recording_url": None,
            "duration": call_data.get("duration"),
            "cost": call_data.get("cost"),
            "status": call_data.get("status"),
            "started_at": call_data.get("startedAt"),
            "ended_at": call_data.get("endedAt"),
            "end_reason": call_data.get("endedReason")
        }
        
        # Extract recording URL
        if "recording" in call_data:
            result["recording_url"] = (
                call_data["recording"].get("stereoUrl") or 
                call_data["recording"].get("url")
            )
        elif "recordingUrl" in call_data:
            result["recording_url"] = call_data["recordingUrl"]
        
        return result
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/user/{user_id}/transcript")
async def get_transcript_by_user_id(user_id: str):
    """
    Get only the transcript for a user
    """
    try:
        obj_id = ObjectId(user_id)
        document = await calls_collection.find_one({"user_id": obj_id})
        
        if not document:
            raise HTTPException(status_code=404, detail="User not found")
        
        call_id = document.get("call_id")
        if not call_id:
            raise HTTPException(status_code=404, detail="Call ID not found")
        
        # Fetch from VAPI
        headers = {"Authorization": f"Bearer {VAPI_TOKEN}"}
        response = requests.get(f"https://api.vapi.ai/call/{call_id}", headers=headers)
        
        if response.status_code == 200:
            transcript = response.json().get("transcript")
            return {
                "user_id": user_id,
                "call_id": call_id,
                "transcript": transcript
            }
        else:
            raise HTTPException(status_code=404, detail="Call not found in VAPI")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/{user_id}/recording")
async def get_recording_by_user_id(user_id: str):
    """
    Get only the recording URL for a user
    """
    try:
        obj_id = ObjectId(user_id)
        document = await calls_collection.find_one({"user_id": obj_id})
        
        if not document:
            raise HTTPException(status_code=404, detail="User not found")
        
        call_id = document.get("call_id")
        if not call_id:
            raise HTTPException(status_code=404, detail="Call ID not found")
        
        # Fetch from VAPI
        headers = {"Authorization": f"Bearer {VAPI_TOKEN}"}
        response = requests.get(f"https://api.vapi.ai/call/{call_id}", headers=headers)
        
        if response.status_code == 200:
            call_data = response.json()
            recording_url = None
            
            if "recording" in call_data:
                recording_url = (
                    call_data["recording"].get("stereoUrl") or 
                    call_data["recording"].get("url")
                )
            elif "recordingUrl" in call_data:
                recording_url = call_data["recordingUrl"]
            
            return {
                "user_id": user_id,
                "call_id": call_id,
                "recording_url": recording_url
            }
        else:
            raise HTTPException(status_code=404, detail="Call not found in VAPI")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class HiringManagerSignup(BaseModel):
    first_name: str
    last_name:str
    email: str
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters long")
    created_at: Optional[datetime] = None
ACCESS_SECRET = os.getenv("JWT_ACCESS_SECRET")
REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

@app.post("/hiring-manager-signup/")
async def hiring_manager_signup(manager: HiringManagerSignup):
    """
    Register a new hiring manager profile in the database.
    Returns the manager ID + tokens upon successful creation.
    """
    logger.info(f"Received hiring manager signup request for: {manager.email}")
    
    try:
        # Prepare data
        manager_dict = manager.dict()
        manager_dict["created_at"] = datetime.utcnow()
        
        # Check if email already exists
        collection = db["hiring_managers"]
        existing_manager = await collection.find_one({"email": manager.email})
        if existing_manager:
            return JSONResponse(
                status_code=400,
                content={"status": False, "message": "Hiring manager with this email already exists"}
            )
        
        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(manager.password.encode('utf-8'), salt)
        manager_dict["password"] = hashed_password.decode('utf-8')
        
        # Insert into DB
        result = await collection.insert_one(manager_dict)
        manager_id = str(result.inserted_id)

        # Generate tokens
        access_token = create_access_token(manager_id, "manager")
        refresh_token = create_refresh_token(manager_id, "manager")

         # save refresh token
        await save_refresh_token(manager_id, "manager", refresh_token)

        return {
            "status": True,
            "message": "Hiring manager profile created successfully",
            "manager_id": manager_id,
            "data": {
                "first_name": manager.first_name,
                "last_name": manager.last_name,
                "email": manager.email
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error creating hiring manager profile: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    

async def get_current_manager(authorization: str = Header(...)):
    
    try:
        token = authorization.split(" ")[1]
        payload = jwt.decode(token, ACCESS_SECRET, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        role: str = payload.get("role")
        if user_id is None or role != "manager":
            raise HTTPException(status_code=401, detail="Invalid token or role")

    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate token")

    # fetch manager from DB
    collection = db["hiring_managers"]
    manager = await collection.find_one({"_id": ObjectId(user_id)})
    if not manager:
        raise HTTPException(status_code=404, detail="Hiring manager not found")

    return manager
class HiringManagerLogin(BaseModel):
    email: str
    password: str

@app.post("/hiring-manager-login/")
async def hiring_manager_login(login_request: HiringManagerLogin):
    """
    Authenticate a hiring manager using email and password.
    Returns access and refresh tokens on successful login.
    """
    try:
        # Find manager by email
        collection = db["hiring_managers"]
        manager = await collection.find_one({"email": login_request.email})
        if not manager:
            return JSONResponse(
                content={"status": False, "message": "Invalid email or password"},
                status_code=401
            )

        # Verify password
        stored_password = manager["password"].encode("utf-8")
        if not bcrypt.checkpw(login_request.password.encode("utf-8"), stored_password):
            return JSONResponse(
                content={"status": False, "message": "Invalid email or password"},
                status_code=401
            )

        manager_id = str(manager["_id"])

        # Generate tokens with role
        access_token = create_access_token(manager_id, role="manager")
        refresh_token = create_refresh_token(manager_id, role="manager")

        # Save refresh token in unified table
        await db["refresh_tokens"].delete_many({"user_id": manager_id, "user_type": "manager"})
        await save_refresh_token(manager_id, user_type="manager", refresh_token=refresh_token)

        return {
            "status": True,
            "message": "Login successful",
            "data": {
                "first_name": manager["first_name"],
                "last_name": manager["last_name"],
                "email": manager["email"],
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except Exception as e:
        error_msg = f"Error during hiring manager login: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    
from fastapi import FastAPI, Query, Body, HTTPException, Header
from enum import Enum

class WorkLocationEnum(str, Enum):
    INPERSON = "inPerson"
    HYBRID = "hybrid"
    REMOTE = "remote"


class JDBasicInfo(BaseModel):
    companyName: str
    jobTitle: str
    roleType: str
    primaryFocus: List[str]
    salesProcessStages: List[str]


class JDExperienceSkills(BaseModel):
    minExp: Optional[int]
    maxExp: Optional[int]
    skillsRequired: List[str]
    workLocation: WorkLocationEnum
    location: List[str]
    timeZone: List[str]


class JDCompensations(BaseModel):
    baseSalary: dict
    ote: List[str]
    equityOffered: bool
    opportunities: List[str]
    keyChallenged: List[str]
    laguages: List[str]


class JDCreationBody(BaseModel):
    basicInfo: JDBasicInfo | None = None
    experienceSkills: JDExperienceSkills | None = None
    compensations: JDCompensations | None = None
    isCompleted: bool = False
    job_id: str | None = None

class JDCreationResponse(BaseModel):
    status: bool
    message: str
    data: dict

# ---------------- Create/Update Job Endpoint ---------------- #
@app.put("/createJob/", response_model=JDCreationResponse)
async def create_job(
    stage: int = Query(..., description="Stage of job creation (1,2,3)"),
    body: JDCreationBody = Body(...),
    authorization: str = Header(...)
):
    """
    Create or update a job role in stages:
      - stage 1 = basicInfo
      - stage 2 = experienceSkills
      - stage 3 = compensations
    Only accessible by hiring managers.
    """
    try:
        # ---------------- Authorization ---------------- #
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        if current_user["role"] != "manager":
            raise HTTPException(status_code=403, detail="Access denied. Only hiring managers can create jobs.")

        collection = db["job_roles"]

        # ---------------- Fetch or Create Job ---------------- #
        if body.job_id:
            try:
                job_oid = ObjectId(body.job_id)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid job_id format")
            job = await collection.find_one({"_id": job_oid})
            if not job:
                raise HTTPException(status_code=404, detail="Job role not found")
        else:
            job = {
                "basicInfo": None,
                "experienceSkills": None,
                "compensations": None,
                "isCompleted": False,
                "created_at": datetime.utcnow(),
                "is_active":True,
                "created_by": current_user["user_id"]  # track creator
            }
            result = await collection.insert_one(job)
            job_oid = result.inserted_id
            job["_id"] = job_oid

        # ---------------- Stage Handling ---------------- #
        update_data = {}
        if stage == 1:
            if not body.basicInfo:
                raise HTTPException(status_code=400, detail="basicInfo is required at stage 1")
            update_data["basicInfo"] = body.basicInfo.dict()

        elif stage == 2:
            if not body.experienceSkills:
                raise HTTPException(status_code=400, detail="experienceSkills is required at stage 2")
            update_data["experienceSkills"] = body.experienceSkills.dict()

        elif stage == 3:
            if not body.compensations:
                raise HTTPException(status_code=400, detail="compensations is required at stage 3")
            update_data["compensations"] = body.compensations.dict()

        else:
            raise HTTPException(status_code=400, detail="Invalid stage. Must be 1, 2, or 3")

        if body.isCompleted:
            update_data["isCompleted"] = True

        # ---------------- Update Mongo Document ---------------- #
        await collection.update_one({"_id": job_oid}, {"$set": update_data})

        updated_job = await collection.find_one({"_id": job_oid})
        updated_job["job_id"] = str(updated_job["_id"])
        updated_job.pop("_id", None)

        return JDCreationResponse(
            status=True,
            message=f"Stage {stage} saved successfully",
            data=updated_job
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error during job creation/update: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/getJob", response_model=JDCreationResponse)
async def get_job(job_id: str = Query(..., description="Job ID to fetch")):
    """
    Fetch a job role from MongoDB using job_id.
    """
    try:
        collection = db["job_roles"]
        
        # Convert string ID to ObjectId
        try:
            job_oid = ObjectId(job_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid job_id format")

        # Fetch document
        job = await collection.find_one({"_id": job_oid})
        if not job:
            raise HTTPException(status_code=404, detail="Job role not found")

        # Convert _id to job_id for response
        job["job_id"] = str(job["_id"])
        job.pop("_id", None)

        return {
            "status": True,
            "message": "Job role fetched successfully",
            "data": job
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error fetching job role: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

# ---------------- Endpoint ---------------- #
# ---------------- Model ---------------- #
class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenResponse(BaseModel):
    status: bool
    message: str
    access_token: str
    token_type: str = "bearer"
    
@app.post("/refresh-access-token/")
async def refresh_access_token(request: RefreshTokenRequest):
    """
    Use a refresh token to generate a new access token.
    Returns JSON responses instead of raising exceptions.
    """
    try:
        # Look up refresh token in DB
        token_doc = await db["refresh_tokens"].find_one({"refresh_token": request.refresh_token})
        if not token_doc:
            return JSONResponse(
                status_code=401,
                content={
                    "status": False,
                    "message": "Invalid or expired refresh token"
                }
            )

        user_id = token_doc["user_id"]
        logger.info(f"Refreshing access token for user_id: {user_id}")
    
        user_type = token_doc["user_type"]
        logger.info(f"User type from token: {user_type}")
        # Generate a new access token
        new_access_token = create_access_token(user_id=user_id, role=user_type)

        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "message": "Access token refreshed successfully",
                "access_token": new_access_token,
                "token_type": "bearer"
            }
        )

    except Exception as e:
        error_msg = f"Error refreshing access token: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "message": error_msg
            }
        )


from datetime import datetime
from fastapi import Header, HTTPException
from fastapi.responses import JSONResponse

@app.get("/my-job-roles/")
async def get_my_job_roles(authorization: str = Header(None)):
    """
    Return all job roles created by the logged-in hiring manager.
    Includes candidate counts for each job role.
    Requires a valid access token in the Authorization header.
    """
    try:
        # Validate token manually
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )

        if current_user["role"] != "manager":
            return JSONResponse(
                content={
                    "status": False,
                    "message": "Access denied. Only hiring managers can access their job roles."
                },
                status_code=403
            )

        job_roles_collection = db["job_roles"]
        profile_collection = db["resume_profiles"]

        cursor = job_roles_collection.find({"created_by": current_user["user_id"]})

        jobs = []
        async for role in cursor:
            role_id = str(role["_id"])

            # Candidate counts
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
                    {"video_email_sent": True},
                    {"application_status": "SendVideoLink"}
                ]
            })

            # Convert ObjectId and serialize
            job = serialize_document(role)
            job["job_id"] = role_id
            job.pop("_id", None)

            # Handle datetime fields safely
            for field in ["created_at", "updated_at"]:
                if field in job and job[field]:
                    if isinstance(job[field], datetime):
                        job[field] = job[field].isoformat()
                    else:
                        job[field] = str(job[field])

            # Append counts
            job.update({
                "total_candidates": total_candidates,
                "audio_attended_count": audio_attended_count,
                "video_attended_count": video_attended_count,
                "moved_to_video_round_count": moved_to_video_round_count
            })

            jobs.append(job)

        return JSONResponse(
            content={
                "status": True,
                "message": "Job roles fetched successfully",
                "data": jobs
            },
            status_code=200
        )

    except HTTPException as he:
        # Catch token issues from get_current_user
        return JSONResponse(
            status_code=he.status_code,
            content={
                "status": False,
                "message": he.detail
            }
        )

    except Exception as e:
        error_msg = f"Error fetching job roles: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "message": error_msg
            }
        )

@app.get("/my-job-candidates/{job_id}")
async def get_my_job_candidates(
    job_id: str,
    authorization: str = Header(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    audio_attended: bool | None = None,
    video_attended: bool | None = None,
    video_interview_sent: bool | None = None,
    application_status: str | None = None,
    seen: bool | None = None,
    shortlisted: bool | None = None,
    call_for_interview: bool | None = None
):
    try:
        # --- Verify access token ---
        if not authorization.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"status": False, "message": "Invalid authorization header"})
        token = authorization.split(" ")[1]

        try:
            payload = verify_access_token(token)
            manager_id = payload.get("sub")
            role = payload.get("role")
            if not manager_id or role != "manager":
                return JSONResponse(status_code=403, content={"status": False, "message": "Unauthorized"})
        except Exception:
            return JSONResponse(status_code=401, content={"status": False, "message": "Invalid or expired token"})

        # --- Validate job belongs to this manager ---
        job_collection = db["job_roles"]
        job = await job_collection.find_one(
            {"_id": ObjectId(job_id), "created_by": manager_id},
            {"basicInfo.jobTitle": 1}
        )
        if not job:
            return JSONResponse(status_code=404, content={"status": False, "message": "Job role not found or not owned by you"})

        # --- Build filters ---
        profile_collection = db["resume_profiles"]
        filter_conditions = {"job_id": job_id}

        # === General filters ===
        if audio_attended is not None:
            filter_conditions["audio_interview"] = True if audio_attended else {"$ne": True}

        if seen is not None:
            filter_conditions["seen_by_manager"] = True if seen else {"$ne": True}

        if application_status is not None:
            if application_status == "SendVideoLink":
                filter_conditions["$or"] = [
                    {"video_email_sent": True},
                    {"application_status": "SendVideoLink"}
                ]
            else:
                filter_conditions["application_status"] = application_status

        if video_attended is not None:
            filter_conditions["video_interview_start"] = True if video_attended else {"$ne": True}

        if video_interview_sent is not None:
            filter_conditions["video_email_sent"] = True if video_interview_sent else {"$ne": True}

        # === Shortlist + Call for Interview logic ===
        if shortlisted is not None:
            if shortlisted:
                if call_for_interview is None:
                    # shortlisted = True, no explicit call_for_interview
                    filter_conditions["final_shortlist"] = True
                    filter_conditions["call_for_interview"] = {"$exists": False}
                else:
                    # shortlisted = True, explicit call_for_interview True/False
                    filter_conditions["final_shortlist"] = True
                    filter_conditions["call_for_interview"] = call_for_interview
            else:
                # shortlisted = False
                filter_conditions["final_shortlist"] = {"$ne": True}

        elif call_for_interview is not None:
            # only call_for_interview filter provided
            filter_conditions["call_for_interview"] = call_for_interview

        # --- Parallel count queries ---
        count_tasks = [
            profile_collection.count_documents(filter_conditions),
            profile_collection.count_documents({**filter_conditions, "audio_interview": True}),
            profile_collection.count_documents({**filter_conditions, "video_interview_start": True}),
            profile_collection.count_documents({
                **filter_conditions,
                "$or": [{"video_email_sent": True}, {"application_status": "SendVideoLink"}]
            })
        ]
        total_candidates, audio_attended_count, video_attended_count, moved_to_video_round_count = await asyncio.gather(*count_tasks)

        total_pages = (total_candidates + page_size - 1) // page_size
        if page > total_pages and total_pages > 0:
            raise HTTPException(status_code=400, detail=f"Page number exceeds total pages ({total_pages})")

        skip = (page - 1) * page_size

        # --- Fetch profiles ---
        profiles = await profile_collection.find(
            filter_conditions,
            {
                "user_id": 1, "application_status": 1, "final_shortlist": 1, "call_for_interview": 1,
                "seen_by_manager": 1, "job_fit_assessment": 1, "audio_updated_job_fit_assessment": 1,
                "audio_interview": 1, "audio_url": 1, "video_url": 1, "video_email_sent": 1,
                "created_at": 1, "processed_video_url": 1, "career_overview": 1
            }
        ).sort("created_at", -1).skip(skip).limit(page_size).to_list(length=page_size)

        if not profiles:
            return {
                "status": True,
                "message": "No candidates found",
                "job_details": {"title": job.get("basicInfo", {}).get("jobTitle", "")},
                "pagination": {"total_candidates": 0, "total_pages": 0},
                "candidates": []
            }

        # === Prefetch and merge user/interview data ===
        user_ids = [ObjectId(p["user_id"]) for p in profiles if p.get("user_id")]
        profile_ids = [str(p["_id"]) for p in profiles]

        user_collection = db["user_accounts"]
        interview_collection = db["interview_sessions"]
        sales_collection = db["sales_scenarios"]
        audio_collection = db["audio_interview_results"]
        audio_proctoring_collection = db["audio_proctoring_logs"]
        video_proctoring_collection = db["video_proctoring_logs"]

        (
            users,
            interviews,
            audio_interviews,
            audio_proc,
            video_proc,
            sales_scenarios
        ) = await asyncio.gather(
            user_collection.find({"_id": {"$in": user_ids}}).to_list(None),
            interview_collection.find({"application_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None),
            audio_collection.find({"application_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None),
            audio_proctoring_collection.find({"user_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None),
            video_proctoring_collection.find({"user_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None),
            sales_collection.find({"user_id": {"$in": profile_ids}}).sort("created_at", -1).to_list(None)
        )

        user_map = {str(u["_id"]): u for u in users}
        interview_map = {i["application_id"]: i for i in interviews}
        audio_map = {a["application_id"]: a for a in audio_interviews}
        audio_proc_map = {p["user_id"]: p for p in audio_proc}
        video_proc_map = {p["user_id"]: p for p in video_proc}
        sales_map = {s["user_id"]: s for s in sales_scenarios}

        candidates = []
        now = datetime.utcnow()

        for profile in profiles:
            user = user_map.get(str(profile.get("user_id")))
            if not user:
                continue

            # --- Compute total experience ---
            career_overview = profile.get("career_overview", {})
            company_history = career_overview.get("company_history", [])
            if company_history:
                company_history.sort(key=lambda x: x.get("start_date", ""), reverse=True)

            total_months = 0
            for role in company_history:
                try:
                    start = datetime.strptime(role.get("start_date", ""), "%Y-%m-%d")
                    if role.get("is_current", False):
                        total_months += (now.year - start.year) * 12 + (now.month - start.month)
                    elif role.get("end_date"):
                        end = datetime.strptime(role["end_date"], "%Y-%m-%d")
                        total_months += (end.year - start.year) * 12 + (end.month - start.month)
                except Exception:
                    continue
            career_overview["total_years_experience"] = round(total_months / 12, 1)

            candidate = {
                "application_id": str(profile["_id"]),
                "profile_created_at": profile.get("created_at").isoformat() if profile.get("created_at") else None,
                "user_id": str(user["_id"]),
                "name": user.get("name", ""),
                "email": user.get("email", ""),
                "phone": user.get("phone", ""),
                "linkedin_url": user.get("linkedin_profile", ""),
                "professional_summary": user.get("professional_summary", ""),
                "basic_information": user.get("basic_information", {}),
                "career_overview": user.get("career_overview", {}),
                "role_process_exposure": user.get("role_process_exposure", {}),
                "sales_context": user.get("sales_context", {}),
                "tools_platforms": user.get("tools_platforms", {}),
                "resume_url": user.get("resume_url", None),
                "application_status": profile.get("application_status", ""),
                "job_fit_assessment": profile.get("job_fit_assessment", ""),
                "audio_updated_job_fit_assessment": profile.get("audio_updated_job_fit_assessment", ""),
                "final_shortlist": profile.get("final_shortlist", False),
                "call_for_interview": profile.get("call_for_interview", False),
                "seen_by_manager": profile.get("seen_by_manager", False),
                "interview_status": {
                    "audio_interview_passed": profile.get("audio_interview", False),
                    "video_interview_attended": bool(profile.get("processed_video_url")),
                    "audio_interview_attended": bool(profile.get("audio_url")),
                    "video_email_sent": profile.get("video_email_sent", False),
                    "video_interview_url": profile.get("video_url"),
                    "processed_video_url": profile.get("processed_video_url", ""),
                    "audio_interview_url": profile.get("audio_url"),
                    "resume_url_from_user_account": user.get("resume_url")
                }
            }

            if interview_map.get(str(profile["_id"])):
                i = interview_map[str(profile["_id"])]
                candidate["interview_details"] = {
                    "session_id": str(i["_id"]),
                    "created_at": i["created_at"].isoformat(),
                    "communication_evaluation": i.get("communication_evaluation", {}),
                    "key_highlights": i.get("interview_highlights", ""),
                    "qa_evaluations": i.get("evaluation", [])
                }

            if audio_map.get(str(profile["_id"])):
                a = audio_map[str(profile["_id"])]
                candidate["audio_interview_details"] = {
                    "audio_interview_id": str(a["_id"]),
                    "created_at": a["created_at"].isoformat(),
                    "qa_evaluations": a.get("qa_evaluations", {}),
                    "audio_interview_summary": a.get("interview_summary", [])
                }

            if audio_proc_map.get(str(profile["_id"])):
                candidate["audio_proctoring_details"] = serialize_document(dict(audio_proc_map[str(profile["_id"])]))

            if video_proc_map.get(str(profile["_id"])):
                candidate["video_proctoring_details"] = serialize_document(dict(video_proc_map[str(profile["_id"])]))

            if sales_map.get(str(profile["_id"])):
                s = sales_map[str(profile["_id"])]
                candidate["sales_scenario_details"] = {
                    "session_id": str(s["_id"]),
                    "created_at": s["created_at"].isoformat(),
                    "sales_conversation_evaluation": s.get("sales_conversation_evaluation", {}),
                    "responses": s.get("responses", [])
                }

            candidates.append(candidate)

        return {
            "status": True,
            "message": "Candidates retrieved successfully",
            "job_details": {
                "title": job["basicInfo"]["jobTitle"],
                "moved_to_video_round_count": moved_to_video_round_count,
                "audio_attended_count": audio_attended_count,
                "video_attended_count": video_attended_count,
                "candidate_count": total_candidates
            },
            "filters": {
                "audio_attended": audio_attended,
                "video_attended": video_attended,
                "application_status": application_status,
                "shortlisted": shortlisted,
                "call_for_interview": call_for_interview
            },
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_candidates": total_candidates,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "candidates": candidates
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error retrieving job candidates: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": False, "message": str(e)})


# --- Request and Response Schemas ---
class ContactUsBody(BaseModel):
    name: str
    designation: str
    companyName: str
    companyEmail: EmailStr
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    query: str | None = None

class ContactUsResponse(BaseModel):
    status: bool
    message: str

# --- Endpoint ---
@app.post("/contact-us/", response_model=ContactUsResponse)
async def contact_us(body: ContactUsBody = Body(...)):
    try:
        collection = db["contact_us"]

        # Insert into DB
        contact_doc = {
            "name": body.name,
            "designation": body.designation,
            "companyName": body.companyName,
            "companyEmail": body.companyEmail,
            "query": body.query,
            "phone": body.phone,
            "linkedin": body.linkedin,
            "created_at": datetime.utcnow()
        }
        await collection.insert_one(contact_doc)
        await notify_admins_for_new_lead(name=body.name, companyName=body.companyName, companyEmail=body.companyEmail, phone=body.phone, query=body.query, linkedin=body.linkedin, designation=body.designation,)
        return JSONResponse(
            content={
                "status": True,
                "message": "Your query has been submitted successfully. Our team will get back to you soon."
            },
            status_code=200
        )
    except Exception as e:
        error_msg = f"Error saving contact-us request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            content={"status": False, "message": error_msg},
            status_code=500
        )
    
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

router = APIRouter()

# --- Schemas ---
class ScheduleInterviewBody(BaseModel):
    applicantName: str
    interviewerName: Optional[str] = None
    jobId: str
    profileId: str
    selectedDate: str
    selectedSlots: List[str]

class ScheduleInterviewResponse(BaseModel):
    status: bool
    message: str
    data: dict

# --- Schedule Interview ---
@app.post("/schedule-interview/", response_model=ScheduleInterviewResponse)
async def schedule_interview(
    body: ScheduleInterviewBody = Body(...),
    authorization: str = Header(None)  # JWT dependency
):
    try:
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        if current_user["role"] != "manager":
            return JSONResponse(
                content={"status": False, "message": "Only managers can schedule interviews"},
                status_code=403
            )

        collection = db["interviews"]

        interview_doc = {
            "applicantName": body.applicantName,
            "interviewerName": body.interviewerName,
            "jobId": body.jobId,
            "profileId": body.profileId,
            "selectedDate": body.selectedDate,
            "selectedSlots": body.selectedSlots,
            "created_by": current_user["user_id"],  # who booked it
            "created_at": datetime.utcnow()
        }

        result = await collection.insert_one(interview_doc)
        interview_id = str(result.inserted_id)

        return JSONResponse(
            content={
                "status": True,
                "message": "Interview scheduled successfully",
                "data": {"interviewId": interview_id}
            },
            status_code=200
        )
    except Exception as e:
        error_msg = f"Error scheduling interview: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(content={"status": False, "message": error_msg}, status_code=500)

# --- Get list of interviews ---
@app.get("/interviews/")
async def list_interviews(
    jobId: Optional[str] = Query(None, description="Filter by Job ID"),
    profileId: Optional[str] = Query(None, description="Filter by Profile ID"),
    authorization: str = Header(None)
):
    try:
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        if current_user["role"] != "manager":
            return JSONResponse(
                content={"status": False, "message": "Only managers can view interviews"},
                status_code=403
            )

        collection = db["interviews"]

        filters = {"created_by": current_user["user_id"]}
        if jobId:
            filters["jobId"] = jobId
        if profileId:
            filters["profileId"] = profileId

        cursor = collection.find(filters).sort("created_at", -1)
        interviews = []
        async for interview in cursor:
            interview["interviewId"] = str(interview["_id"])
            interview.pop("_id", None)
            interviews.append(serialize_document(interview))

        return JSONResponse(
            content={
                "status": True,
                "message": "Interviews fetched successfully",
                "data": interviews
            },
            status_code=200
        )
    except Exception as e:
        error_msg = f"Error fetching interviews: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(content={"status": False, "message": error_msg}, status_code=500)

# --- Get details of an interview ---
@app.get("/interviews/{interview_id}")
async def get_interview_details(interview_id: str, authorization: str = Header(None)):
    try:
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        if current_user["role"] != "manager":
            return JSONResponse(
                content={"status": False, "message": "Only managers can view interview details"},
                status_code=403
            )

        collection = db["interviews"]

        interview = await collection.find_one({"_id": ObjectId(interview_id), "created_by": current_user["user_id"]})
        if not interview:
            return JSONResponse(
                content={"status": False, "message": "Interview not found or not owned by you"},
                status_code=404
            )

        interview["interviewId"] = str(interview["_id"])
        interview.pop("_id", None)
        interview = serialize_document(interview)
        return JSONResponse(
            content={
                "status": True,
                "message": "Interview details fetched successfully",
                "data": interview
            },
            status_code=200
        )
    except Exception as e:
        error_msg = f"Error fetching interview details: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(content={"status": False, "message": error_msg}, status_code=500)

###new candidate apis
class CadidateSignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: str 
    canidate_source: str
    linkedin_profile: Optional[str] = None

@app.post("/candidate-signup/")
async def candidate_signup(body: CadidateSignupRequest):
    try:
        profile_dict = body.dict()
        profile_dict["created_at"] = datetime.utcnow()
        collection = db["user_accounts"]
        # Check if email already exists
        existing_candidate = await collection .find_one({"email": body.email})
        if existing_candidate:
            return JSONResponse(
                content={"status": False, "message": "Email already registered"},
                status_code=400
            )
        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(body.password.encode('utf-8'), salt)
        profile_dict["password"] = hashed_password.decode('utf-8')
        result= await collection.insert_one(profile_dict)
        candidate_id = str(result.inserted_id)
        # Generate tokens
        access_token = create_access_token(candidate_id, "candidate")
        refresh_token = create_refresh_token(candidate_id, "canidate")
        await save_refresh_token(candidate_id, "candidate", refresh_token)
        return JSONResponse(
            content={
                "status": True,
                "message": "Candidate registered successfully",
                "data": {"candidate_id": candidate_id},
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            },
            status_code=200
        )
    except Exception as e:
        error_msg = f"Error during candidate signup: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(content={"status": False, "message": error_msg}, status_code=500)

class CandidateLoginRequest(BaseModel):
    email: EmailStr
    password: str

# @app.post("/candidate-login/")
# async def candidate_login(body: CandidateLoginRequest):
#     try:
#         collection = db["user_accounts"]
#         candidate = await collection.find_one({"email": body.email})
#         if not candidate:
#             return JSONResponse(
#                 content={"status": False, "message": "Invalid email or password"},
#                 status_code=401
#             )
#         # Verify password
#         stored_password = candidate["password"].encode("utf-8")
#         if not bcrypt.checkpw(body.password.encode("utf-8"), stored_password):
#             return JSONResponse(
#                 content={"status": False, "message": "Invalid email or password"},
#                 status_code=401
#             )

#         candidate_id = str(candidate["_id"])

#         # Generate tokens with role
#         access_token = create_access_token(candidate_id, role="candidate")
#         refresh_token = create_refresh_token(candidate_id, role="candidate")

#         # Save refresh token in unified table
#         await db["refresh_tokens"].delete_many({"user_id": candidate_id, "user_type": "candidate"})
#         await save_refresh_token(candidate_id, user_type="candidate", refresh_token=refresh_token)

#         return {
#             "status": True,
#             "message": "Login successful",
#             "data": {
#                 "candidate_id": candidate_id,
#                 "name": candidate["name"],
#                 "email": candidate["email"],
#             },
#             "access_token": access_token,
#             "refresh_token": refresh_token,
#             "token_type": "bearer"
#         }

#     except Exception as e:
#         error_msg = f"Error during hiring manager login: {str(e)}"
#         logger.error(error_msg, exc_info=True)
#         raise HTTPException(status_code=500, detail=error_msg)
async def evaluate_audio_interview(application_id: str):
    """
    Evaluate interview Q&A pairs stored in audio_interview_results for the given application_id.
    Updates evaluation results and resume_profiles accordingly.
    """

    try:
        collection = db["resume_profiles"]
        audio_results_collection = db["audio_interview_results"]

        # Fetch audio_interview_results document
        audio_doc = await audio_results_collection.find_one({"application_id": application_id})
        if not audio_doc:
            return {"status": False, "message": "Audio interview results not found"}

        # user_id = audio_doc.get("application_id")
        # if not user_id:
        #     return {"status": False, "message": "User ID missing in audio interview results"}

        profile = await collection.find_one({"_id": ObjectId(application_id)})
        if not profile:
            return {"status": False, "message": "Resume profile not found"}

        job_fit_assessment = profile.get("job_fit_assessment", "")

        results = []
        scores = []
        strengths = []
        areas_for_improvement = []
        red_flags = []
        sales_motions = []
        sales_cycles = []
        icp_entries = []
        coaching_focus = []

        # Loop over questionAnswers (including probe if present)
        for idx, qa_pair in enumerate(audio_doc.get("questionAnswers", [])):
            i = idx + 1
            question = qa_pair.get("question")
            answer = qa_pair.get("answer")
            probe = qa_pair.get("probe")

            if not question or not answer:
                continue

            evaluation_result = await evaluate_answer(question, answer, i)

            results.append({
                "question": question,
                "answer": answer,
                "evaluation": evaluation_result.dict()
            })

            logger.info(f"Evaluated Q{i}")

            if hasattr(evaluation_result, "score"):
                scores.append(evaluation_result.score)

            if evaluation_result.score >= (0.75 * (20 if i == 1 else 15 if i == 2 else 5 if i == 3 else 100)):
                strengths.append(evaluation_result.fit_summary)

            if evaluation_result.score <= (0.4 * (20 if i == 1 else 15 if i == 2 else 5 if i == 3 else 100)):
                areas_for_improvement.append(evaluation_result.fit_summary)

            if evaluation_result.red_flags:
                red_flags.extend(evaluation_result.red_flags)

            if evaluation_result.sales_motion != "not mentioned":
                sales_motions.append(evaluation_result.sales_motion)

            if evaluation_result.sales_cycle != "not mentioned":
                sales_cycles.append(evaluation_result.sales_cycle)

            if evaluation_result.icp:
                icp_entries.append(evaluation_result.icp)

            if evaluation_result.coaching_focus:
                coaching_focus.append(evaluation_result.coaching_focus)

            # If probe exists and has Q&A, evaluate that too
            if probe and probe.get("question") and probe.get("answer"):
                probe_eval = await evaluate_answer(probe["question"], probe["answer"], i)
                results.append({
                    "question": probe["question"],
                    "answer": probe["answer"],
                    "evaluation": probe_eval.dict()
                })

                if hasattr(probe_eval, "score"):
                    scores.append(probe_eval.score)

        # Compute averages
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0
        normalized_scores = []
        for idx, score in enumerate(scores):
            q_num = idx + 1
            max_score = 20 if q_num == 1 else 15 if q_num == 2 else 5 if q_num == 3 else 100
            normalized_scores.append((score / max_score) * 100)

        avg_normalized = round(sum(normalized_scores) / len(normalized_scores), 2) if normalized_scores else 0
        audio_interview_status = avg_normalized >= 65

        # Generate OTP
        import string, random
        otp = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

        # Update resume profile
        await collection.update_one(
            {"_id": ObjectId(application_id)},
            {"$set": {
                "audio_interview": True,
                "interview_otp": otp,
                "video_interview_start": False
            }}
        )

        # Update audio_interview_results document with evaluation
        evaluation_doc = {
            "qa_evaluations": results,
            "interview_summary": {
                "average_score": avg_score,
                "average_normalized_score": avg_normalized,
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
            "evaluated_at": datetime.utcnow()
        }

        await audio_results_collection.update_one(
            {"_id": audio_doc["_id"]},
            {"$set": evaluation_doc}
        )

        # Update job fit assessment
        audio_updated_job_fit_assessment = await audio_updated_fit_assessment(
            job_fit_assessment=job_fit_assessment,
            audio_interview=str(evaluation_doc)
        )
        await collection.update_one(
            {"_id": ObjectId(application_id)},
            {"$set": {"audio_updated_job_fit_assessment": audio_updated_job_fit_assessment}}
        )

        return {
            "status": True,
            "message": "Audio interview evaluated successfully.",
            "qualified_for_video_round": audio_interview_status
        }

    except Exception as e:
        error_msg = f"Error evaluating audio interview: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    
@app.post("/my-manager-profile/")
async def get_manager_profile(authorization: str = Header(...)):
    try:
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        if current_user["role"] != "manager":
            return JSONResponse(
                content={"status": False, "message": "Only managers can access this endpoint"},
                status_code=403
            )
        manager_id = current_user["user_id"]
        if not manager_id:
            raise HTTPException(status_code=400, detail="Invalid token payload")    
        manager = await db["hiring_managers"].find_one({"_id": ObjectId(manager_id)})
        if not manager:
            raise HTTPException(status_code=404, detail="Manager not found")
        job_roles_collection = db["job_roles"]
        profile_collection = db["resume_profiles"]

        cursor = job_roles_collection.find({"created_by": current_user["user_id"]})

        jobs = []
        async for role in cursor:
            role_id = str(role["_id"])

            # Candidate counts
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
                    {"video_email_sent": True},
                    {"application_status": "SendVideoLink"}
                ]
            })

            # Convert ObjectId and serialize
            job = serialize_document(role)
            job["job_id"] = role_id
            job.pop("_id", None)

            # Handle datetime fields safely
            for field in ["created_at", "updated_at"]:
                if field in job and job[field]:
                    if isinstance(job[field], datetime):
                        job[field] = job[field].isoformat()
                    else:
                        job[field] = str(job[field])

            # Append counts
            job.update({
                "total_candidates": total_candidates,
                "audio_attended_count": audio_attended_count,
                "video_attended_count": video_attended_count,
                "moved_to_video_round_count": moved_to_video_round_count
            })

            jobs.append(job)
        manager_profile = {
            "manager_id": str(manager["_id"]),
            "first_name": manager.get("first_name", ""),
            "last_name": manager.get("last_name", ""),
            "email": manager.get("email", ""),
            "profile_created_at": manager.get("created_at").isoformat() if manager.get("created_at") else None,
            "jobs":jobs
        }

        return {
            "status": True,
            "message": "Manager profile retrieved successfully",
            "data": manager_profile
        }
    except HTTPException as he:
        raise he
    except Exception as e:  
        error_msg = f"Error retrieving manager profile: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/candidate-login/")
async def candidate_login(body: CandidateLoginRequest):
    try:
        collection = db["user_accounts"]
        candidate = await collection.find_one({"email": body.email})
        if not candidate:
            return JSONResponse(
                content={"status": False, "message": "Invalid email or password"},
                status_code=401
            )
        # Verify password
        if "password" not in candidate:
            return JSONResponse(
                content={"status": False, "message": '''Oops! Looks like you haven't set a password yet. Click "Forgot Password" below to set one up.'''},
                status_code=401
            )
        stored_password = candidate["password"].encode("utf-8")
   
            
        if not bcrypt.checkpw(body.password.encode("utf-8"), stored_password):
            return JSONResponse(
                content={"status": False, "message": "Invalid email or password"},
                status_code=401
            )

        candidate_id = str(candidate["_id"])

        # Generate tokens with role
        access_token = create_access_token(candidate_id, role="candidate")
        refresh_token = create_refresh_token(candidate_id, role="candidate")

        # Save refresh token in unified table
        await db["refresh_tokens"].delete_many({"user_id": candidate_id, "user_type": "candidate"})
        await save_refresh_token(candidate_id, user_type="candidate", refresh_token=refresh_token)

        # --- Fetch all applications for this candidate ---
        resume_collection = db["resume_profiles"]
        job_collection = db["job_roles"]

        applications_cursor = resume_collection.find({"user_id": candidate_id})
        applications = await applications_cursor.to_list(length=None)

        # Collect all job_ids to fetch jobs in bulk
        job_ids = list({app.get("job_id") for app in applications if app.get("job_id")})
        jobs_cursor = job_collection.find({"_id": {"$in": [ObjectId(jid) for jid in job_ids]}})
        jobs = await jobs_cursor.to_list(length=None)

        # Create a map job_id -> jobTitle
        job_map = {str(job["_id"]): job.get("basicInfo", {}).get("jobTitle", "Unknown") for job in jobs}

        # Prepare application history
        application_history = []
        for app in applications:
            job_id = app.get("job_id")
            job_role_name = job_map.get(job_id, "Unknown")
            application_history.append({
                "application_id": str(app["_id"]),
                "job_role_name": job_role_name,
                "job_id": job_id,
                "application_status": app.get("application_status", ""),
                "video_interview_start": app.get("video_interview_start", False),
                "video_email_sent": app.get("video_email_sent", False),
                "audio_interview_status": app.get("audio_interview", False),
            })

        return {
            "status": True,
            "message": "Login successful",
            "data": {
                "candidate_id": candidate_id,
                "name": candidate["name"],
                "email": candidate["email"],
                "application_history": application_history
            },
            "is_profile_complete":{
                "resume_uploaded": bool(candidate.get("resume_text")),
                "profile_information": bool(candidate.get("basic_information")),
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except Exception as e:
        error_msg = f"Error during candidate login: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

from fastapi import Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from bson import ObjectId
from datetime import datetime

@app.get("/my-candidate-profile/")
async def get_candidate_profile(authorization: str = Header(...)):
    """
    Fetch candidate profile and application history using access token.
    """
    try:
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        candidate_id = current_user["user_id"]
        if not candidate_id:
            raise HTTPException(status_code=400, detail="Invalid token payload")

        # --- Fetch candidate account ---
        candidate = await db["user_accounts"].find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # --- Fetch all applications for this candidate ---
        resume_collection = db["resume_profiles"]
        job_collection = db["job_roles"]

        applications_cursor = resume_collection.find({"user_id": candidate_id})
        applications = await applications_cursor.to_list(length=None)

        # Collect all job_ids to fetch jobs in bulk
        job_ids = list({app.get("job_id") for app in applications if app.get("job_id")})
        jobs_cursor = job_collection.find({"_id": {"$in": [ObjectId(jid) for jid in job_ids]}})
        jobs = await jobs_cursor.to_list(length=None)

        # Create a map job_id -> jobTitle
        job_map = {str(job["_id"]): job.get("basicInfo", {}).get("jobTitle", "Unknown") for job in jobs}

        # Prepare application history
        application_history = []
        for app in applications:
            job_id = app.get("job_id")
            job_role_name = job_map.get(job_id, "Unknown")
            application_history.append({
                "application_id": str(app["_id"]),
                "job_role_name": job_role_name,
                "job_id": job_id,
                "application_status": app.get("application_status", ""),
                "video_interview_start": app.get("video_interview_start", False),
                "video_email_sent": app.get("video_email_sent", False),
                "audio_interview_status": app.get("audio_interview", False),
            })

        return {
            "status": True,
            "message": "Candidate profile fetched successfully",
            "data": {
                "candidate_id": candidate_id,
                "name": candidate.get("name", ""),
                "email": candidate.get("email", ""),
                "phone": candidate.get("phone", ""),
                "basic_information": candidate.get("basic_information", {}),
                "career_overview": candidate.get("career_overview", {}),
                "role_process_exposure": candidate.get("role_process_exposure", {}),
                "sales_context": candidate.get("sales_context", {}),
                "tools_platforms": candidate.get("tools_platforms", {}),
                "resume_url": candidate.get("resume_url", None),
                "application_history": application_history
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error fetching candidate profile: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
@app.post("/candidate-set-password/")
async def candidate_set_password(
    new_password: str = Body(..., embed=True, min_length=6),
    email:str= Body(..., embed=True),
    otp: str = Body(..., embed=True,)
):
    try:
        collection = db["user_accounts"]
        candidate = await collection.find_one({"email": email})
        if not candidate:
            return JSONResponse(
                content={"status": False, "message": "Email not found"},
                status_code=404
            )
        acc_otp= candidate.get("otp")
        if acc_otp != otp:
            return JSONResponse(
                content={"status": False, "message": "Invalid OTP"},
                status_code=400
            )
        # Ensure timezone consistency
        otp_time = candidate.get("otp_created_at")
        if otp_time.tzinfo is None:
            otp_time = otp_time.replace(tzinfo=timezone.utc)

        current_time = datetime.now(timezone.utc)
        elapsed_seconds = (current_time - otp_time).total_seconds()

        if elapsed_seconds > 20 * 60:  # 20 minutes
            return JSONResponse(
                content={"status": False, "message": "OTP has expired"},
                status_code=400
            )
        # Hash the new password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)
        # Update password in DB
        await collection.update_one(
            {"_id": candidate["_id"]},
            {"$set": {"password": hashed_password.decode('utf-8'), "updated_at": datetime.utcnow()}}
        )
        return JSONResponse(
            content={"status": True, "message": "Password updated successfully"},
            status_code=200
        )
    except Exception as e:
        error_msg = f"Error setting new password: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(content={"status": False, "message": error_msg}, status_code=500)


@app.post("/parse-candidate-resume/")
async def parse_candidate_resume(
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    """
    Parse a resume and update resume_url & resume_text
    in the logged-in user's account.
    """
    try:
        # Validate token manually
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        if isinstance(current_user, JSONResponse):
            return current_user  # return token error as JSON

        user_id = current_user["user_id"]

        logger.info(f"Received resume parsing request from user {user_id} for file: {file.filename}")

        # Read file content
        content = await file.read()
        if not content:
            return JSONResponse(
                status_code=400,
                content={"status": False, "message": "File not readable"}
            )

        # Upload to Azure Blob Storage
        resume_url = None
        if file.filename.lower().endswith(".pdf"):
            file_for_blob = UploadFile(
                filename=file.filename,
                file=io.BytesIO(content)
            )
            resume_url, _ = await upload_to_blob_storage_resume(file_for_blob, "resume")
            logger.info(f"Resume uploaded to blob: {resume_url}")

        # Extract text (PyPDF → fallback OCR)
        text = ""
        try:
            text = await extract_text_from_pdf(
                UploadFile(filename=file.filename, file=io.BytesIO(content))
            )
            if not text:
                logger.info("PyPDF returned empty text, falling back to OCR")
                text = await extract_text_with_ocr(
                    UploadFile(filename=file.filename, file=io.BytesIO(content))
                )
        except Exception as e:
            logger.warning(f"PDF/OCR extraction failed: {str(e)}")
            text = ""

        # Update user_accounts for this user
        user_accounts = db["user_accounts"]
        await user_accounts.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"resume_url": resume_url, "resume_text": text,"updated_at": datetime.utcnow()}},
            upsert=False
        )

        logger.info(f"Updated resume for user {user_id}")
        parsed_data = await parse_resume_with_azure(text)
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "message": "Resume uploaded and parsed successfully",
                "resume_url": resume_url,
                "data": ResumeData(**parsed_data).dict()
            }
        )

    except HTTPException as e:
        logger.error(f"HTTP error during resume parsing: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=e.status_code,
            content={"status": False, "message": str(e.detail)}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": "Internal server error"}
        )
class CandidateBio(BaseModel):
    #user_id: str
    job_id: Optional[str] = None
    basic_information: BasicInformation
    career_overview: CareerOverview
    sales_context: SalesContext
    role_process_exposure: RoleProcessExposure
    tools_platforms: OptionalToolsPlatforms
@app.post("/update-candidate-data/")
async def update_candidate_data(
    profile: CandidateBio,
    authorization: str = Header(...)
):
    """
    Update candidate data for the logged-in user.
    - Uses access token to identify the user.
    - Updates user_accounts with profile details.
    - Creates a new resume_profiles record linked to the user.
    - Adds resume_profile_id into user_accounts.application_ids.
    """
    try:
        # Validate token and get current user
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        user_id = current_user["user_id"]

        profile_dict = profile.dict()
        # job_id = profile_dict.get("job_id")

        # # Validate job_id if provided
        # if job_id:
        #     job_collection = db["job_roles"]
        #     job = await job_collection.find_one({"_id": ObjectId(job_id)})
        #     if not job:
        #         return JSONResponse(status_code=404, content={"status": False, "message": "Job role not found"})

        # Step 1: Update user_accounts with profile details
        user_accounts = db["user_accounts"]
        update_fields = {
            "basic_information": profile_dict.get("basic_information"),
            "career_overview": profile_dict.get("career_overview"),
            "sales_context": profile_dict.get("sales_context"),
            "role_process_exposure": profile_dict.get("role_process_exposure"),
            "tools_platforms": profile_dict.get("tools_platforms"),
            "updated_at": datetime.utcnow()
        }
        await user_accounts.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_fields}
        )
        #user_doc = await user_accounts.find_one({"_id": ObjectId(user_id)})
    #     resume_text = user_doc.get("resume_text", "") if user_doc else ""
    #     job_description= job if job else {}
    #     job_fit_assessment = await generate_job_fit_summary(resume_text, job_description)
    #     # Step 2: Insert into resume_profiles
    #     resume_profiles = db["resume_profiles"]
    #     new_profile_doc = {
    #     "user_id": user_id,
    #     "job_id": job_id,
    #     "video_url": None,
    #     "video_uploaded_at": None,
    #     "audio_url": None,
    #     "job_fit_assessment": job_fit_assessment,
    #     "audio_uploaded_at": None,
    #     "created_at": datetime.utcnow(),
    #     "updated_at": datetime.utcnow()
    # }
    #     result = await resume_profiles.insert_one(new_profile_doc)
    #     application_id = str(result.inserted_id)

        # # Step 3: Append application_id into user_accounts.application_ids
        # await user_accounts.update_one(
        #     {"_id": ObjectId(user_id)},
        #     {"$push": {"application_ids": application_id}}
        # )

        return {
            "status": True,
            "message": "Candidate data updated successfully",
            #"application_id": application_id
        }

    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": False, "message": he.detail}
        )

    except Exception as e:
        error_msg = f"Error updating candidate data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(status_code=500, content={"status": False, "message": error_msg})

@app.post("/update-professional-summary/")
async def update_professional_summary(
    professional_summary: Dict = Body(...),
    authorization: str = Header(...)
):
    """
    Update the professional summary for the logged-in candidate.
    """
    try:
        # Get current user from JWT
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        user_id = current_user["user_id"]

        # Update professional_summary field in user_accounts
        result = await db["user_accounts"].update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "professional_summary": professional_summary,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        if result.matched_count == 0:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "User not found"}
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "message": "Professional summary updated successfully"
            }
        )

    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": False, "message": he.detail}
        )

    except Exception as e:
        error_msg = f"Error updating professional summary: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": error_msg}
        )
class RemindLaterRequest(BaseModel):
    application_id: str
    remaind_at: str
from brevo_mail import send_remind_later_email, send_password_reset_email, notify_admins_for_new_lead
@app.post("/remind-later/")
async def remind_me_later(
    request: RemindLaterRequest,
    authorization: str = Header(...)
):
    """
    Update the 'remind me later' date for a user's resume profile.
    Requires a valid access token.
    """
    try:
        # Verify token and get current user
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        user_id = current_user["user_id"]
        # Fetch full user info from user_accounts
        user_doc = await db["user_accounts"].find_one({"_id": ObjectId(user_id)})
        if not user_doc:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "User account not found"}
            )

        candidate_name = user_doc.get("name", "Unknown Candidate")
        candidate_phone = user_doc.get("phone", "N/A")
        candidate_email = user_doc.get("email", "N/A")
        collection = db["resume_profiles"]

        # Find resume profile by application_id and user_id
        resume_profile = await collection.find_one({
            "_id": ObjectId(request.application_id),
            "user_id": user_id
        })

        if not resume_profile:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Resume profile not found for this user"}
            )

        # Update fields
        update_data = {
            "remaind_later": True,
            "remaind_at": request.remaind_at,
            "updated_at": datetime.utcnow(),
            "remaind_status_updated_at": datetime.utcnow()
        }

        await collection.update_one(
            {"_id": ObjectId(request.application_id)},
            {"$set": update_data}
        )
        # utc_dt = datetime.fromisoformat(request.remaind_at)
        # ist_tz = pytz.timezone("Asia/Kolkata")
        # ist_dt = utc_dt.astimezone(ist_tz)
        remind_at_str = request.remaind_at
        send_remind_later_email(candidate_name, remind_at_str, candidate_email, candidate_phone)
        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "message": "Remind me later updated successfully",
                "application_id": request.application_id,
                "data": {
                "remaind_at": request.remaind_at
            }}
        )

    except HTTPException as he:
        return JSONResponse(
            status_code=he.status_code,
            content={"status": False, "message": he.detail}
        )

    except Exception as e:
        error_msg = f"Error updating remind me later: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": error_msg}
        )

# async def call_llm(messages: list[dict]):
#     async with aiohttp.ClientSession() as session:
#         async with session.post(
#             AZURE_OPENAI_URL,
#             headers=AZURE_HEADERS,
#             json={
#                 "messages": messages,
#                 "temperature": 0.7,
#                 "max_tokens": 600
#             }
#         ) as response:
#             if response.status != 200:
#                 raise HTTPException(status_code=500, detail=f"LLM error {response.status}")
#             result = await response.json()
#             return result["choices"][0]["message"]["content"].strip()
# # Collections
# user_accounts = db["user_accounts"]
# resume_profiles = db["resume_profiles"]
# audio_results = db["audio_interview_results_collection"]

# # Request model
# class AudioInterviewCallRequest(BaseModel):
#     answer: str | None = None
#     application_id: str

# # --- Decide probing using LLM ---
# async def needs_probing(question: str, answer: str) -> bool:
#     probe_prompt = f"""
# You are an expert interviewer.
# Question: {question}
# Candidate Answer: {answer}

# Decide if probing is needed. Reply only "YES" or "NO".
# """
#     result = await call_llm([
#         {"role": "system", "content": "You are a strict interviewer."},
#         {"role": "user", "content": probe_prompt}
#     ])
#     return "YES" in result.upper()

# # --- Generate interview questions (Q1, Q2, Q3) ---
# async def generate_interview_questions(resume_text: str):
#     prompt = f"""
# You are an expert recruiter.
# Generate exactly 3 interview questions from this resume:

# {resume_text}

# Rules:
# Q1: Performance/Achievement
# Q2: Industry/Buyer Persona
# Q3: Motivation/Future Fit

# Follow the exact templates provided earlier.
# Return only a valid Python list of 3 strings.
# """
#     result = await call_llm([
#         {"role": "system", "content": "You are a professional interviewer."},
#         {"role": "user", "content": prompt}
#     ])
#     try:
#         return eval(result)
#     except:
#         raise HTTPException(status_code=500, detail="Invalid LLM output format")

# # --- Main Interview Endpoint ---
# @app.post("/audio-interview/")
# async def audio_interview(
#     request: AudioInterviewCallRequest,
#     authorization: str = Header(...)
# ):
#     try:
#         # Use your existing JWT verification
#         current_user = await get_current_user(authorization)
#         user_id = current_user["user_id"]

#         # fetch user account (resume_text lives here)
#         user_doc = await user_accounts.find_one({"_id": ObjectId(user_id)})
#         if not user_doc:
#             return JSONResponse({"status": False, "message": "User not found"}, status_code=404)

#         resume_text = user_doc.get("resume_text", "")
#         if not resume_text:
#             return JSONResponse({"status": False, "message": "Resume not found"}, status_code=400)

#         # find or create session for this application
#         session = await audio_results.find_one({"user_id": str(user_id), "application_id": request.application_id})
#         if not session:
#             questions = await generate_interview_questions(resume_text)
#             session = {
#                 "user_id": str(user_id),
#                 "application_id": request.application_id,
#                 "resume_text": resume_text,
#                 "questions": questions,
#                 "questions_asked": [],
#                 "answers": [],
#                 "created_at": datetime.utcnow(),
#                 "completed": False
#             }
#             await audio_results.insert_one(session)

#         # reload session
#         session = await audio_results.find_one({"user_id": str(user_id), "application_id": request.application_id})
#         asked = session["questions_asked"]
#         answers = session["answers"]
#         questions = session["questions"]

#         # if first call → ask Q1
#         if not asked:
#             first_q = questions[0]
#             asked.append(first_q)
#             await audio_results.update_one({"_id": session["_id"]}, {"$set": {"questions_asked": asked}})
#             return {"status": True, "message": "First question", "question": first_q, "done": False}

#         # otherwise → store answer
#         if request.answer is None:
#             raise HTTPException(status_code=400, detail="Answer required")

#         answers.append(request.answer)
#         await audio_results.update_one({"_id": session["_id"]}, {"$set": {"answers": answers}})

#         # current Q is last asked
#         current_q = asked[-1]

#         # Q1 & Q2 → check probing
#         if len(asked) < 3:
#             probe_needed = await needs_probing(current_q, request.answer)
#             if probe_needed and not session.get("probing_done", False):
#                 probe_q = "That's great to hear! Just to help me understand the scope better — and specifically your role in achieving that?"
#                 await audio_results.update_one({"_id": session["_id"]}, {"$set": {"probing_done": True}})
#                 return {"status": True, "message": "Probe", "question": probe_q, "done": False}

#         # if no probe → next question
#         if len(asked) < len(questions):
#             next_q = questions[len(asked)]
#             asked.append(next_q)
#             await audio_results.update_one(
#                 {"_id": session["_id"]},
#                 {"$set": {"questions_asked": asked, "probing_done": False}}
#             )
#             return {"status": True, "message": "Next question", "question": next_q, "done": False}

#         # if Q3 answered → complete interview
#         await audio_results.update_one(
#             {"_id": session["_id"]},
#             {"$set": {"completed": True, "completed_at": datetime.utcnow()}}
#         )

#         # mark application profile as audio_interview = True
#         await resume_profiles.update_one(
#             {"_id": ObjectId(request.application_id)},
#             {"$set": {"audio_interview": True}}
#         )

#         return {"status": True, "message": "Interview completed", "done": True}

#     except Exception as e:
#         error_msg = f"Error in audio interview: {str(e)}"
#         logger.error(error_msg, exc_info=True)
#         raise HTTPException(status_code=500, detail=error_msg)

class InterviewQuestionRequest(BaseModel):
    application_id: str

@app.post("/audio-interview-questions/")
async def audio_interview_questions(
        request: InterviewQuestionRequest,
    authorization: str = Header(...)
):
    """
    Generate interview questions based on stored resume data and job requirements.
    Uses application_id instead of profile_id.
    Gets posting title from job_roles collection (basicInfo.jobTitle).
    """
    try:
        # ✅ Verify JWT
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )

        if not current_user or "user_id" not in current_user:
            return JSONResponse(
                content={"status": False, "message": "Unauthorized"},
                status_code=401
            )

        user_id = current_user["user_id"]

        # ✅ Use application_id to fetch resume profile
        resume_collection = db["resume_profiles"]
        profile = await resume_collection.find_one({"_id": ObjectId(request.application_id)})

        if not profile:
            return JSONResponse(
                content={"message": "Resume profile not found", "status": False},
                status_code=200
            )

        if "audio_interview" in profile:
            return JSONResponse(
                content={"message": "Audio round already completed", "status": False},
                status_code=200
            )

        resume_text = profile.get("resume_text", "")

        questions = await generate_audio_intv_questions(
            resume_text=resume_text,
            
        )

        return {"status": True,
                "message": "Questions generated successfully",
                "data": {"questions": questions}
                }

    except Exception as e:
        error_msg = f"Error generating questions: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(content={"status": False, "message": error_msg}, status_code=500)

class InterviewEvaluation(BaseModel):
    qa_pairs: List[QAPair]
    application_id: str
@app.post("/evaluate-audio-interview/")
async def evaluate_interview(
    evaluation: InterviewEvaluation,
    authorization: str = Header(...)
):
    """
    Evaluate interview Q&A pairs and provide detailed feedback.
    Also updates the audio interview status in the database based on average score.
    """
    logger.info("Received interview evaluation request")
    try:
        # ✅ Verify JWT
        try:
            current_user = await get_current_user(authorization)
        except HTTPException:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )

        if not current_user or "user_id" not in current_user:
            return JSONResponse(
                content={"status": False, "message": "Unauthorized"},
                status_code=401
            )

        user_id = current_user["user_id"]

        # ✅ Fetch resume profile using application_id from payload
        application_id = evaluation.application_id
        collection = db["resume_profiles"]
        audio_results_collection = db["audio_interview_results"]

        profile = await collection.find_one({"_id": ObjectId(application_id)})
        job_fit_assessment = profile.get("job_fit_assessment", "") if profile else ""

        if not profile:
            return JSONResponse(content={"message": "Resume profile not found"}, status_code=404)

        # ✅ Fetch candidate details from user_accounts using user_id from token
        user_accounts_collection = db["user_accounts"]
        candidate_info = await user_accounts_collection.find_one({"_id": ObjectId(user_id)})

        if not candidate_info:
            return JSONResponse(content={"message": "Candidate not found"}, status_code=404)

        results = []
        scores = []
        strengths = []
        areas_for_improvement = []
        red_flags = []
        sales_motions = []
        sales_cycles = []
        icp_entries = []
        coaching_focus = []

        for idx, qa_pair in enumerate(evaluation.qa_pairs):
            i = idx + 1
            evaluation_result = await evaluate_answer(qa_pair.question, qa_pair.answer, i)

            results.append({
                "question": qa_pair.question,
                "answer": qa_pair.answer,
                "evaluation": evaluation_result.dict()
            })
            logger.info(f"Evaluated Q{i}")

            if hasattr(evaluation_result, "score"):
                scores.append(evaluation_result.score)

            if evaluation_result.score >= (0.75 * (20 if i == 1 else 15 if i == 2 else 5 if i == 3 else 100)):
                strengths.append(evaluation_result.fit_summary)

            if evaluation_result.score <= (0.4 * (20 if i == 1 else 15 if i == 2 else 5 if i == 3 else 100)):
                areas_for_improvement.append(evaluation_result.fit_summary)

            if evaluation_result.red_flags:
                red_flags.extend(evaluation_result.red_flags)

            if evaluation_result.sales_motion != "not mentioned":
                sales_motions.append(evaluation_result.sales_motion)

            if evaluation_result.sales_cycle != "not mentioned":
                sales_cycles.append(evaluation_result.sales_cycle)

            if evaluation_result.icp:
                icp_entries.append(evaluation_result.icp)

            if evaluation_result.coaching_focus:
                coaching_focus.append(evaluation_result.coaching_focus)

        avg_score = round(sum(scores) / len(scores), 2) if scores else 0

        normalized_scores = []
        for idx, score in enumerate(scores):
            q_num = idx + 1
            max_score = 20 if q_num == 1 else 15 if q_num == 2 else 5 if q_num == 3 else 100
            normalized_scores.append((score / max_score) * 100)

        avg_normalized = round(sum(normalized_scores) / len(normalized_scores), 2) if normalized_scores else 0
        audio_interview_status = avg_normalized >= 65

        import string, random
        otp = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

        # ✅ Update resume profile
        await collection.update_one(
            {"_id": ObjectId(application_id)},
            {"$set": {
                "audio_interview": True,
                "interview_otp": otp,
                "video_interview_start": False
            }}
        )

        # ✅ Store detailed evaluation
        evaluation_doc = {
            "user_id": user_id,
            "application_id": application_id,
            "qa_evaluations": results,
            "interview_summary": {
                "average_score": avg_score,
                "average_normalized_score": avg_normalized,
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

        audio_updated_job_fit_assessment = await audio_updated_fit_assessment(
            job_fit_assessment=job_fit_assessment,
            audio_interview=str(evaluation_doc)
        )

        await collection.update_one(
            {"_id": ObjectId(application_id)},
            {"$set": {"audio_updated_job_fit_assessment": audio_updated_job_fit_assessment}}
        )

        return {
            "status": True,
            "message": "Thank you for completing the audio interview. Your answers have been recorded.",
            "qualified_for_video_round": audio_interview_status
        }

    except Exception as e:
        error_msg = f"Error evaluating interview: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/video-interview/")
async def conversational_interview(
    application_id: str = Form(None),
    session_id: str = Form(None),
    user_answer: str = Form(None),
    authorization: str = Header(...)
):
    """
    Stateful, stepwise conversational interview endpoint.
    - First call: application_id + token, returns session_id and first question.
    - Subsequent calls: session_id + user_answer, returns next question or closing prompt.
    - Final: saves final thoughts and marks interview completed.
    """
    # --- Verify JWT ---
    try:
        current_user = await get_current_user(authorization)
    except HTTPException:
        return JSONResponse(
            content={"status": False, "message": "Invalid or expired token"},
            status_code=401
        )

    if not current_user or "user_id" not in current_user:
        return JSONResponse(
            content={"status": False, "message": "Unauthorized"},
            status_code=401
        )

    candidate_user_id = current_user["user_id"]

    collection = db["interview_sessions"]
    resume_collection = db["resume_profiles"]
    user_collection = db["user_accounts"]  # fetch resume text from here
    job_config = None

    # --- First call: application_id + token ---
    if application_id is not None and session_id is None:
        # Fetch resume profile using application_id
        user_profile = await resume_collection.find_one({"_id": ObjectId(application_id)})
        if not user_profile:
            raise HTTPException(status_code=404, detail="Resume profile not found")

        # Check candidate matches token
        if user_profile.get("user_id") != candidate_user_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to this application")
        job_id = user_profile.get("job_id")
        if not job_id:
            return JSONResponse(
            status_code=200,
            content={"status": False, "message": "Job ID not found in application"}
        )
        invite_status= user_profile.get("video_email_sent", False)
        logger.info(f"Video interview invite status: {invite_status}")
        if not invite_status:
            return JSONResponse(
            status_code=200,
            content={"status": False, "message": "not invited for video interview"}
        )
        if user_profile.get("video_interview_start", False):
            return JSONResponse(
            status_code=200,
            content={"status": False, "message": "Video interview already started"}
        )
        
        
        
        job_config = get_job_config_by_job_id(job_id)
        if not job_config:
            job_id="68dbb0e6e07e4078863fcf7b"
            job_config = get_job_config_by_job_id(job_id)
            #raise HTTPException(status_code=404, detail=f"Job configuration not found for job ID: {job_id}")

        interview_questions = job_config["interview_questions"]
        role_from_config = job_config["job_role"]

        # --- Fetch resume_text from user_accounts ---
        user_account = await user_collection.find_one({"_id": ObjectId(candidate_user_id)})
        if not user_account or "resume_text" not in user_account:
            raise HTTPException(status_code=400, detail="resume_text missing in user account")
        resume_text = user_account["resume_text"]

        # Create session
        session = {
            "application_id": application_id,
            "user_id": candidate_user_id,
            "role": role_from_config,
            "job_id": job_id,
            "resume_text": resume_text,
            "current_question": 0,
            "answers": [],
            "created_at": datetime.utcnow()
        }
        result = await collection.insert_one(session)
        session_id = str(result.inserted_id)

        if not interview_questions:
            raise HTTPException(status_code=500, detail="No questions configured for this job ID.")
        reset_count = user_profile.get("video_interview_reset_count", 0)
        processed_video_url=f'https://scooterdata.blob.core.windows.net/scooter-processed-videos/{application_id}_video_{reset_count}_master.m3u8'
        first_question = interview_questions[0]
        await collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"last_question": first_question["question"]}}
        )
        await resume_collection.update_one(
            {"_id": ObjectId(application_id)},
            {"$set": {"video_interview_start": True, "processed_video_url": processed_video_url}}
        )
        reset_count= user_profile.get("video_interview_reset_count", 0)
        return {
            "session_id": session_id,
            "question": first_question["question"],
            "step": "question",
            "total_questions": len(interview_questions)+1,
            "reset_count": reset_count
        }

    # --- Subsequent calls: session_id + user_answer ---
    if session_id is not None:
        session = await collection.find_one({"_id": ObjectId(session_id)})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Ensure candidate owns this session
        if session.get("user_id") != candidate_user_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to session")

        stored_job_id = session.get("job_id")
        if not stored_job_id:
            raise HTTPException(status_code=500, detail="Job ID missing from session")

        job_config = get_job_config_by_job_id(stored_job_id)
        if not job_config:
            raise HTTPException(status_code=404, detail=f"Job config not found for job ID: {stored_job_id}")

        interview_questions = job_config["interview_questions"]
        role = session["role"]
        resume_text = session["resume_text"]
        current_question_idx = session["current_question"]
        answers = session["answers"]
        last_question = session.get("last_question")

        # Final closing prompt
        if current_question_idx >= len(interview_questions) and user_answer is not None:
            final_thoughts = {
                "question_number": len(interview_questions) + 1,
                "question": "Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?",
                "answer": user_answer,
                "timestamp": datetime.utcnow()
            }
            answers.append(final_thoughts)
            await collection.update_one({"_id": ObjectId(session_id)}, {"$set": {"answers": answers}})
            evaluate_commu = await evaluate_communication(session_id)
            evaluation_result = await evaluate_intervieww(session_id, answers, resume_text, role, job_config)
            key_highlights = await generate_key_highlights(session_id)
            await collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {
                    "evaluation": evaluation_result,
                    "interview_completed": True,
                    "completed_at": datetime.utcnow()
                }}
            )
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
            closing_prompt = "Is there anything else you'd like to share — about how you work, what motivates you, or why this role excites you?"
            await collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"current_question": len(interview_questions)}}
            )
            return {
                "session_id": session_id,
                "question": closing_prompt,
                "step": "done"
            }

    raise HTTPException(status_code=400, detail="Invalid request. Must provide application_id (first call) or session_id+user_answer (subsequent calls).")

class CandidateApplicationStatusRequest(BaseModel):
    application_id: str
    application_status: str
    reason: Optional[str] = None

@app.post("/candidate-application-status/")
async def update_application_status(
    request: CandidateApplicationStatusRequest,
    authorization: str = Header(...)
):
    """
    Update the application status for a user in the resume profile collection.
    Only the hiring manager who created the job can update the status.
    """
    try:
        # --- Verify JWT and get current user ---
        current_user = await get_current_user(authorization)
        if not current_user or current_user.get("role") != "manager":
            return JSONResponse(
                status_code=403,
                content={"status": False, "message": "Unauthorized"}
            )

        manager_id = current_user["user_id"]

        # --- Fetch resume profile using application_id ---
        collection = db["resume_profiles"]
        profile = await collection.find_one({"_id": ObjectId(request.application_id)})
        if not profile:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Resume profile not found"}
            )

        job_id = profile.get("job_id")
        if not job_id:
            return JSONResponse(
                status_code=400,
                content={"status": False, "message": "Job ID missing in resume profile"}
            )

        # --- Fetch job and validate manager ownership ---
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Job not found"}
            )
        if job.get("created_by") != manager_id:
            return JSONResponse(
                status_code=403,
                content={"status": False, "message": "You are not authorized to update this application"}
            )

        # --- Update the application status ---
        update_data = {
            "application_status": request.application_status,
            "application_status_updated_at": datetime.utcnow()
        }
        if request.reason:
            update_data["application_status_reason"] = request.reason

        await collection.update_one(
            {"_id": ObjectId(request.application_id)},
            {"$set": update_data}
        )

        return {
            "status": True,
            "message": "Application status updated successfully",
            "application_id": request.application_id,
            "application_status": request.application_status,
            "reason": request.reason
        }

    except Exception as e:
        error_msg = f"Error updating application status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    
class CandidateFinalShortlistRequest(BaseModel):
    application_id: str
    final_shortlist: bool
    reason: Optional[str] = None    

@app.post("/candidate-final-shortlist/")
async def shortlist_application_status(
    request: CandidateFinalShortlistRequest,
    authorization: str = Header(...)
):
    """
    Update the final shortlist status for a user in the resume profile collection.
    Only the hiring manager who created the job can update the shortlist.
    """
    try:
        current_user = await get_current_user(authorization)
        if not current_user or current_user.get("role") != "manager":
            return JSONResponse(
                status_code=403,
                content={"status": False, "message": "Unauthorized"}
            )

        manager_id = current_user["user_id"]

        # --- Fetch resume profile using application_id ---
        collection = db["resume_profiles"]
        profile = await collection.find_one({"_id": ObjectId(request.application_id)})
        if not profile:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Resume profile not found"}
            )

        job_id = profile.get("job_id")
        if not job_id:
            return JSONResponse(
                status_code=400,
                content={"status": False, "message": "Job ID missing in resume profile"}
            )

        # --- Fetch job and validate manager ownership ---
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Job not found"}
            )
        if job.get("created_by") != manager_id:
            return JSONResponse(
                status_code=403,
                content={"status": False, "message": "You are not authorized to update this shortlist"}
            )

        # --- Update the final shortlist status ---
        update_data = {
            "final_shortlist": request.final_shortlist,
            "shortlist_status_updated_at": datetime.utcnow()
        }
        if request.reason:
            update_data["shortlist_status_reason"] = request.reason

        await collection.update_one(
            {"_id": ObjectId(request.application_id)},
            {"$set": update_data}
        )

        return {
            "status": True,
            "message": "Shortlist status updated successfully",
            "application_id": request.application_id,
            "shortlist_status": request.final_shortlist,
            "reason": request.reason
        }

    except Exception as e:
        error_msg = f"Error updating shortlist status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
class ResetVideoInterviewRequest(BaseModel):
    application_id: str
    reset_reason: Optional[str] = None

@app.post("/reset-candidate-video-interview/")
async def reset_video_interview(
    request: ResetVideoInterviewRequest,
    authorization: str = Header(...)
):
    """
    Reset the video interview status for an application.
    """
    try:
        current_user = await get_current_user(authorization)
        if not current_user or current_user.get("role") != "manager":
            return JSONResponse(
                status_code=403,
                content={"status": False, "message": "Unauthorized"}
            )
        manager_id = current_user["user_id"]

        collection = db["resume_profiles"]

        # Fetch the application using application_id
        application = await collection.find_one({"_id": ObjectId(request.application_id)})
        if not application:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Application not found"}
            )

        job_id = application.get("job_id")
        if not job_id:
            return JSONResponse(
                status_code=400,
                content={"status": False, "message": "Job ID missing in application"}
            )

        # Verify that the manager owns this job
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id), "created_by": manager_id})
        if not job:
            return JSONResponse(
                status_code=403,
                content={"status": False, "message": "You are not authorized to reset this application's video interview"}
            )

        # Reset video interview fields
        update_data = {
            "video_interview_start": False,
            "video_url": None,
            "video_uploaded_at": None,
            "video_interview_reset_reason": request.reset_reason,
            "video_interview_reset_at": datetime.utcnow()
        }

        await collection.update_one(
            {"_id": ObjectId(request.application_id)},
            {
                "$set": update_data,
                "$inc": {"video_interview_reset_count": 1}
            }
        )

        return {
            "status": True,
            "message": "Video interview reset successfully",
            "application_id": request.application_id,
            "reset_reason": request.reset_reason
        }

    except Exception as e:
        error_msg = f"Error resetting video interview: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": error_msg}
        )
async def call_llm(messages: list[dict]):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 600
            }
        ) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail=f"LLM error {response.status}")
            result = await response.json()
            logger.info(f"LLM response: {result}")
            return result["choices"][0]["message"]["content"].strip()
async def needs_probing(question: str, answer: str) -> bool:
    probe_prompt = f"""
You are an expert interviewer.
Question: {question}
Candidate Answer: {answer}

Decide if probing is needed based on the clarity of the answer. Reply only "YES" or "NO".
"""
    result = await call_llm([
        {"role": "system", "content": "You are a strict interviewer."},
        {"role": "user", "content": probe_prompt}
    ])
    return "YES" in result.upper()
# --- Generate probing follow-ups ---
def get_probe(question_index: int) -> str:
    if question_index == 0:  # Q1 probe
        return "If someone else were trying to replicate your success, what would you tell them to focus on?"
    elif question_index == 1:  # Q2 probe
        return "If you had to sum it up, what’s the key to winning over that kind of buyer or handling their objections?"
    else:
        return None
user_accounts = db["user_accounts"]
resume_profiles = db["resume_profiles"]
audio_results = db["audio_interview_results"]
def get_next_question_response(qa_list, questions):
    next_q = questions[len(qa_list)]
    is_last = len(qa_list) + 1 == len(questions)
    qa_list.append({"question": next_q, "answer": None, "probe": None})
    stage = "done" if is_last else "question"
    message = "Last question" if is_last else "Next question"
    return qa_list, {"status": True, "message": message, "question": next_q, "done": False, "stage": stage}
class AudioInterviewTempCallRequest(BaseModel):
    answer: str | None = None
    application_id: str
# --- Main Interview Endpoint ---
@app.post("/candidate-audio-interview/")
async def audio_interview(
    request: AudioInterviewTempCallRequest,
    authorization: str = Header(...)
):
    try:
        # Validate token
        try:
            current_user = await get_current_user(authorization)
        except HTTPException:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )

        user_id = current_user["user_id"]

        # Fetch user
        user_doc = await user_accounts.find_one({"_id": ObjectId(user_id)})
        if not user_doc:
            return JSONResponse({"status": False, "message": "User not found"}, status_code=404)

        resume_text = user_doc.get("resume_text", "")
        if not resume_text:
            return JSONResponse({"status": False, "message": "Resume not found"}, status_code=400)

        # Find session
        session = await audio_results.find_one({"user_id": str(user_id), "application_id": request.application_id})

        # ✅ CASE 1: Already completed
        if session and session.get("completed", False):
            return JSONResponse(
                {
                    "status": False,
                    "message": "You have already completed this interview.",
                    "done": True,
                    "stage": "done"
                },
                status_code=200
            )

        # ✅ CASE 2: If session exists and user tries to start again (no answer)
        if session and not session.get("completed", False) and request.answer is None:
            qa_list = session.get("questionAnswers", [])
            if len(qa_list) > 0:
                # Interview already started — block restarting
                return JSONResponse(
                    {
                        "status": True,
                        "message": "You have already started this interview.",
                        "done": False,
                        "stage": "in_progress"
                    },
                    status_code=200
                )

        # ✅ CASE 3: Create session if not exists (new interview)
        if not session:
            questions = await generate_audio_intv_questions(resume_text)
            session = {
                "user_id": str(user_id),
                "application_id": request.application_id,
                "resume_text": resume_text,
                "questions": questions,
                "questionAnswers": [],
                "created_at": datetime.utcnow(),
                "completed": False,
                "probing_stage": False
            }
            await audio_results.insert_one(session)

        # Reload session
        session = await audio_results.find_one({"user_id": str(user_id), "application_id": request.application_id})
        qa_list = session.get("questionAnswers", [])
        questions = session["questions"]

        # --- NORMAL FLOW FROM HERE (unchanged) ---

        def get_next_question_response(qa_list, questions):
            next_q = questions[len(qa_list)]
            is_last = len(qa_list) + 1 == len(questions)
            qa_list.append({"question": next_q, "answer": None, "probe": None})
            stage = "done" if is_last else "question"
            message = "Last question" if is_last else "Next question"
            return qa_list, {"status": True, "message": message, "question": next_q, "done": False, "stage": stage}

        # CASE 1: first call → ask Q1
        if not qa_list:
            qa_list, response = get_next_question_response(qa_list, questions)
            await audio_results.update_one(
                {"_id": session["_id"]},
                {"$set": {"questionAnswers": qa_list, "probing_stage": False}}
            )
            return response

        # CASE 2: answer required
        if request.answer is None:
            raise HTTPException(status_code=400, detail="Answer required")

        # get the last QA entry
        current_entry = qa_list[-1]

        # If probing stage → save probe answer and move forward
        if session.get("probing_stage", False):
            if current_entry.get("probe"):
                current_entry["probe"]["answer"] = request.answer
            else:
                current_entry["probe"] = {"question": None, "answer": request.answer}
            await audio_results.update_one(
                {"_id": session["_id"]},
                {"$set": {"questionAnswers": qa_list, "probing_stage": False}}
            )
            # move to next question if available
            if len(qa_list) < len(questions):
                qa_list, response = get_next_question_response(qa_list, questions)
                await audio_results.update_one({"_id": session["_id"]}, {"$set": {"questionAnswers": qa_list}})
                return response
            else:
                # All questions answered → complete interview
                await audio_results.update_one(
                    {"_id": session["_id"]},
                    {"$set": {"completed": True, "completed_at": datetime.utcnow()}}
                )
                await resume_profiles.update_one(
                    {"_id": ObjectId(request.application_id)},
                    {"$set": {"audio_interview": True}}
                )
                temp = await evaluate_audio_interview(request.application_id)
                logger.info(f"Audio interview evaluated: {temp}")
                return {"status": True, "message": "Interview completed", "done": True, "stage": "done"}

        # Normal Q answer (not probing)
        if current_entry["answer"] is None:
            current_entry["answer"] = request.answer
        await audio_results.update_one({"_id": session["_id"]}, {"$set": {"questionAnswers": qa_list}})

        current_q_index = len(qa_list) - 1
        current_q = current_entry["question"]

        # Check probing for Q1/Q2
        if current_q_index < 2:
            probe_needed = await needs_probing(current_q, request.answer)
            if probe_needed:
                probe_q = get_probe(current_q_index)
                current_entry["probe"] = {"question": probe_q, "answer": None}
                await audio_results.update_one(
                    {"_id": session["_id"]},
                    {"$set": {"questionAnswers": qa_list, "probing_stage": True}}
                )
                return {"status": True, "message": "Probe", "question": probe_q, "done": False, "stage": "question"}

        # Next question (normal flow)
        if len(qa_list) < len(questions):
            qa_list, response = get_next_question_response(qa_list, questions)
            await audio_results.update_one({"_id": session["_id"]}, {"$set": {"questionAnswers": qa_list}})
            return response

        # Complete interview if all answered
        await audio_results.update_one(
            {"_id": session["_id"]},
            {"$set": {"completed": True, "completed_at": datetime.utcnow()}}
        )
        await resume_profiles.update_one(
            {"_id": ObjectId(request.application_id)},
            {"$set": {"audio_interview": True}}
        )
        temp = await evaluate_audio_interview(request.application_id)
        logger.info(f"Audio interview evaluated: {temp}")
        return {"status": True, "message": "Interview completed", "done": True, "stage": "done"}

    except Exception as e:
        error_msg = f"Error in audio interview: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class ApplyJobRequest(BaseModel):
    job_id: str

@app.post("/apply-job/")
async def apply_job(
    body: ApplyJobRequest,
    authorization: str = Header(...)
):
    """
    Endpoint for candidates to apply for a job.
    Validates candidate's access token, checks job existence,
    generates job fit assessment, and stores application.
    """
    try:
        try:
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        user_id = current_user["user_id"]

        job_id = body.job_id
        resume_profiles = db["resume_profiles"]
        user_accounts = db["user_accounts"]
        # --- Step 2: Validate job existence ---
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Job not found"}
            )

        # --- Step 3: Fetch user details ---
        
        user_doc = await user_accounts.find_one({"_id": ObjectId(user_id)})
        if not user_doc:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "User not found"}
            )

        resume_text = user_doc.get("resume_text", "")
        job_description = job if job else {}
        application_ids = user_doc.get("application_ids", [])
        if application_ids:
            # Instead of calling DB again, use aggregation pipeline to check in one go
            pipeline = [
                {"$match": {"_id": {"$in": [ObjectId(aid) for aid in application_ids]}}},
                {"$match": {"job_id": job_id}},
                {"$limit": 1}
            ]
            cursor = resume_profiles.aggregate(pipeline)
            if await cursor.to_list(length=1):
                return JSONResponse(
                    status_code=400,
                    content={"status": False, "applied":True,"message": "You already have an active application for this job"},
                )
        if not resume_text or not user_doc.get("basic_information"):
            return JSONResponse(
        status_code=400,
        content={
            "status": False,
            "message": "Complete your profile with resume and basic information before applying.",
            "is_profile_complete":{
                "resume_uploaded": bool(user_doc.get("resume_text")),
                "profile_information": bool(user_doc.get("basic_information")),
            }
        }
    )
        # --- Step 4: Generate job fit assessment ---
        job_fit_assessment = await generate_job_fit_summary(resume_text, job_description)

        # --- Step 5: Insert application into resume_profiles ---
        
        new_profile_doc = {
            "user_id": user_id,
            "job_id": job_id,
            "video_url": None,
            "video_uploaded_at": None,
            "audio_url": None,
            "audio_uploaded_at": None,
            "job_fit_assessment": job_fit_assessment,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await resume_profiles.insert_one(new_profile_doc)
        application_id = str(result.inserted_id)

        # --- Step 6: Add application_id to user_accounts.application_ids ---
        await user_accounts.update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"application_ids": application_id}}
        )

        return {
            "status": True,
            "message": "Job application submitted successfully",
            "applied": True,
            "application_id": application_id,
            "job_id": job_id,
        }

    except Exception as e:
        logger.error(f"Error in /apply-job/: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": f"Internal server error: {str(e)}"}
        )

@app.post("/reset-password/")
async def reset_password(
    body: dict,
):
    """
    Endpoint to initiate password reset process.
    Generates a reset token, stores it, and sends a reset email.
    """
    try:
        user_collection = db["user_accounts"]
        email = body.get("email")
        user = await user_collection.find_one({"email": email})
        if not user:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Email not found"}
            )
        otp= otp = ''.join(random.choices(string.digits, k=6))
        user_name=user.get("name", "User")
        user_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"otp": otp, "otp_created_at": datetime.utcnow()}}
        )
        temp= await send_password_reset_email(email, otp, user_name)
        logger.info(f"Password reset email sent: {temp}")
        return {
            "status": True,
            "message": "Password reset email sent"
        }
    except Exception as e:
        error_msg = f"Error in /reset-password/: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": error_msg}
        )

class AdminSignup(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    phone: Optional[str] = None

@app.post("/admin-signup/")
async def hiring_manager_signup(admin:AdminSignup):
    """
    Register a new hiring manager profile in the database.
    Returns the manager ID + tokens upon successful creation.
    """
    logger.info(f"Received hiring manager signup request for: {admin.email}")
    
    try:
        # Prepare data
        admin_dict = admin.dict()
        admin_dict["created_at"] = datetime.utcnow()
        
        # Check if email already exists
        collection = db["admin_accounts"]
        existing_manager = await collection.find_one({"email": admin.email})
        if existing_manager:
            return JSONResponse(
                status_code=400,
                content={"status": False, "message": "admin with this email already exists"}
            )
        
        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(admin.password.encode('utf-8'), salt)
        admin_dict["password"] = hashed_password.decode('utf-8')
        
        # Insert into DB
        result = await collection.insert_one(admin_dict)
        super_admin_id = str(result.inserted_id)

        # Generate tokens
        access_token = create_access_token(super_admin_id, "superadmin")
        refresh_token = create_refresh_token(super_admin_id, "superadmin")

         # save refresh token
        await save_refresh_token(super_admin_id, "superadmin", refresh_token)

        return {
            "status": True,
            "message": "Admin profile created successfully",
            "admin_id": super_admin_id,
            "data": {
                "first_name": admin.first_name,
                "last_name": admin.last_name,
                "email": admin.email
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        error_msg = f"Error creating admin profile: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class AdminLogin(BaseModel):
    email: str
    password: str

@app.post("/admin-login/")
async def hiring_manager_login(login_request: AdminLogin):
    """
    Authenticate a hiring manager using email and password.
    Returns access and refresh tokens on successful login.
    """
    try:
        # Find manager by email
        collection = db["admin_accounts"]
        admin = await collection.find_one({"email": login_request.email})
        if not admin:
            return JSONResponse(
                content={"status": False, "message": "Invalid email or password"},
                status_code=401
            )

        # Verify password
        stored_password = admin["password"].encode("utf-8")
        if not bcrypt.checkpw(login_request.password.encode("utf-8"), stored_password):
            return JSONResponse(
                content={"status": False, "message": "Invalid email or password"},
                status_code=401
            )

        admin_id = str(admin["_id"])

        # Generate tokens with role
        access_token = create_access_token(admin_id, role="superadmin")
        refresh_token = create_refresh_token(admin_id, role="superadmin")

        # Save refresh token in unified table
        await db["refresh_tokens"].delete_many({"user_id": admin_id, "user_type": "superadmin"})
        await save_refresh_token(admin_id, user_type="superadmin", refresh_token=refresh_token)

        return {
            "status": True,
            "message": "Login successful",
            "data": {
                "first_name": admin["first_name"],
                "last_name": admin["last_name"],
                "email": admin["email"],
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    except Exception as e:
        error_msg = f"Error during admin login: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    
@app.post("/admin-reset-password/")
async def reset_password(
    body: dict,
):
    """
    Endpoint to initiate password reset process.
    Generates a reset token, stores it, and sends a reset email.
    """
    try:
        user_collection = db["admin_accounts"]
        email = body.get("email")
        user = await user_collection.find_one({"email": email})
        if not user:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Email not found"}
            )
        otp= otp = ''.join(random.choices(string.digits, k=6))
        user_name=user.get("name", "User")
        user_collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"otp": otp, "otp_created_at": datetime.utcnow()}}
        )
        temp= await send_password_reset_email(email, otp, user_name)
        logger.info(f"Password reset email sent: {temp}")
        return {
            "status": True,
            "message": "Password reset email sent"
        }
    except Exception as e:
        error_msg = f"Error in /reset-password/: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": error_msg}
        )
@app.post("/admin-set-password/")
async def candidate_set_password(
    new_password: str = Body(..., embed=True, min_length=6),
    email:str= Body(..., embed=True),
    otp: str = Body(..., embed=True,)
):
    try:
        collection = db["admin_accounts"]
        candidate = await collection.find_one({"email": email})
        if not candidate:
            return JSONResponse(
                content={"status": False, "message": "Email not found"},
                status_code=404
            )
        acc_otp= candidate.get("otp")
        if acc_otp != otp:
            return JSONResponse(
                content={"status": False, "message": "Invalid OTP"},
                status_code=400
            )
        # Ensure timezone consistency
        otp_time = candidate.get("otp_created_at")
        if otp_time.tzinfo is None:
            otp_time = otp_time.replace(tzinfo=timezone.utc)

        current_time = datetime.now(timezone.utc)
        elapsed_seconds = (current_time - otp_time).total_seconds()

        if elapsed_seconds > 20 * 60:  # 20 minutes
            return JSONResponse(
                content={"status": False, "message": "OTP has expired"},
                status_code=400
            )
        # Hash the new password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)
        # Update password in DB
        await collection.update_one(
            {"_id": candidate["_id"]},
            {"$set": {"password": hashed_password.decode('utf-8'), "updated_at": datetime.utcnow()}}
        )
        return JSONResponse(
            content={"status": True, "message": "Password updated successfully"},
            status_code=200
        )
    except Exception as e:
        error_msg = f"Error setting new password: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(content={"status": False, "message": error_msg}, status_code=500)
class UserRequest(BaseModel):
    user_id: str
def serialize_document(doc):
    """Recursively convert MongoDB document to JSON-serializable dict"""
    if not doc:
        return {}
    def convert(value):
        if isinstance(value, ObjectId):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, list):
            return [convert(v) for v in value]
        elif isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        else:
            return value
    return convert(doc)


@app.post("/get-user-details")
async def get_user_details(body: UserRequest,authorization: str = Header(...),):
    """
    Get full user details with all applications (and linked data) for a user_id.
    Includes job title from 'job_roles' collection.
    """
    try:
        if not authorization.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"status": False, "message": "Invalid authorization header"})
        token = authorization.split(" ")[1]

        try:
            payload = verify_access_token(token)
            admin_id = payload.get("sub")
            role = payload.get("role")
            if not admin_id or role != "superadmin":
                return JSONResponse(status_code=403, content={"status": False, "message": "Unauthorized"})
        except Exception:
            return JSONResponse(status_code=401, content={"status": False, "message": "Invalid or expired token"})
        user_id = body.user_id
        user_collection = db["user_accounts"]
        interview_collection = db["interview_sessions"]
        sales_collection = db["sales_scenarios"]
        audio_collection = db["audio_interview_results"]
        audio_proc_collection = db["audio_proctoring_logs"]
        video_proc_collection = db["video_proctoring_logs"]
        job_roles = db["job_roles"] 
        if not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        # 1️⃣ Fetch user
        user = await user_accounts.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 2️⃣ Fetch all profiles (applications)
        profiles = await resume_profiles.find({"user_id": user_id}).to_list(None)
        if not profiles:
            return JSONResponse(
                content={"status": True, "data": {"user": serialize_document(user), "applications": []}}
            )

        # Collect all application IDs and job IDs
        application_ids = [str(p["_id"]) for p in profiles]
        job_ids = [
            ObjectId(p["job_id"])
            for p in profiles
            if p.get("job_id") and ObjectId.is_valid(p["job_id"])
        ]

        # 3️⃣ Bulk fetch all linked documents
        interview_docs = await interview_collection.find({"application_id": {"$in": application_ids}}).to_list(None)
        audio_docs = await audio_collection.find({"application_id": {"$in": application_ids}}).to_list(None)
        audio_proc_docs = await audio_proc_collection.find({"application_id": {"$in": application_ids}}).to_list(None)
        video_proc_docs = await video_proc_collection.find({"application_id": {"$in": application_ids}}).to_list(None)
        sales_docs = await sales_collection.find({"application_id": {"$in": application_ids}}).to_list(None)
        job_docs = await job_roles.find(
            {"_id": {"$in": job_ids}}, {"basicInfo.jobTitle": 1}
        ).to_list(None)  # <-- fetch only jobTitle field

        # 4️⃣ Create lookup maps
        interview_map = {doc["application_id"]: doc for doc in interview_docs}
        audio_map = {doc["application_id"]: doc for doc in audio_docs}
        audio_proc_map = {doc["application_id"]: doc for doc in audio_proc_docs}
        video_proc_map = {doc["application_id"]: doc for doc in video_proc_docs}
        sales_map = {doc["application_id"]: doc for doc in sales_docs}
        job_map = {str(doc["_id"]): doc for doc in job_docs}

        applications = []

        # 5️⃣ Merge data for each profile
        for profile in profiles:
            application_id = str(profile["_id"])
            job_id = str(profile.get("job_id")) if profile.get("job_id") else None
            job_doc = job_map.get(job_id)

            app_data = {
                "application_id": application_id,
                "profile_created_at": profile.get("created_at").isoformat() if profile.get("created_at") else None,
                "application_status": profile.get("application_status", ""),
                "final_shortlist": profile.get("final_shortlist", False),
                "call_for_interview": profile.get("call_for_interview", False),
                "job_details": {
                    "job_id": job_id,
                    "job_title": job_doc.get("basicInfo", {}).get("jobTitle") if job_doc else None
                },
                "interview_status": {
                    "audio_interview_passed": profile.get("audio_interview", False),
                    "video_interview_attended": bool(profile.get("processed_video_url")),
                    "audio_interview_attended": bool(profile.get("audio_url")),
                    "video_email_sent": profile.get("video_email_sent", False),
                    "video_interview_url": profile.get("video_url"),
                    "processed_video_url": profile.get("processed_video_url", ""),
                    "audio_interview_url": profile.get("audio_url"),
                    "resume_url_from_user_account": user.get("resume_url")
                }
            }

            # 🔹 Attach all linked data
            if interview_map.get(application_id):
                i = interview_map[application_id]
                app_data["interview_details"] = serialize_document({
                    "session_id": i["_id"],
                    "created_at": i.get("created_at"),
                    "communication_evaluation": i.get("communication_evaluation", {}),
                    "key_highlights": i.get("interview_highlights", ""),
                    "qa_evaluations": i.get("evaluation", [])
                })

            if audio_map.get(application_id):
                a = audio_map[application_id]
                app_data["audio_interview_details"] = serialize_document({
                    "audio_interview_id": a["_id"],
                    "created_at": a.get("created_at"),
                    "qa_evaluations": a.get("qa_evaluations", {}),
                    "audio_interview_summary": a.get("interview_summary", [])
                })

            if audio_proc_map.get(application_id):
                app_data["audio_proctoring_details"] = serialize_document(audio_proc_map[application_id])

            if video_proc_map.get(application_id):
                app_data["video_proctoring_details"] = serialize_document(video_proc_map[application_id])

            if sales_map.get(application_id):
                s = sales_map[application_id]
                app_data["sales_scenario_details"] = serialize_document({
                    "session_id": s["_id"],
                    "created_at": s.get("created_at"),
                    "sales_conversation_evaluation": s.get("sales_conversation_evaluation", {}),
                    "responses": s.get("responses", [])
                })

            applications.append(app_data)

        # 6️⃣ Combine everything
        result = {
            "user": serialize_document({
                "_id": user["_id"],
                "name": user.get("name", ""),
                "email": user.get("email", ""),
                "phone": user.get("phone", ""),
                "professional_summary": user.get("professional_summary", ""),
                "basic_information": user.get("basic_information", {}),
                "career_overview": user.get("career_overview", {}),
                "role_process_exposure": user.get("role_process_exposure", {}),
                "sales_context": user.get("sales_context", {}),
                "tools_platforms": user.get("tools_platforms", {}),
                "resume_url": user.get("resume_url", None),
            }),
            "applications": applications
        }

        return JSONResponse(content={"status": True, "data": result})

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": f"Server error: {str(e)}"}
        )
class JobStatusUpdateRequest(BaseModel):
    job_id: str
    new_status: bool
@app.post("/job-status/")
async def job_status(
    body: JobStatusUpdateRequest,
    authorization: str = Header(...)
):
    """
    Update the status of a job.
    Validates manager's access token and ownership of the job.
    """
    try:
        # 1️⃣ Validate user
        try:
            current_user = await get_current_user(authorization)
        except HTTPException:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        manager_id = current_user["user_id"]

        # 2️⃣ Validate job_id
        if not ObjectId.is_valid(body.job_id):
            return JSONResponse(
                content={"status": False, "message": "Invalid job ID"},
                status_code=400
            )

        job_collection = db["job_roles"]

        # 3️⃣ Check if job exists and belongs to manager
        job = await job_collection.find_one({"_id": ObjectId(body.job_id), "created_by": manager_id})
        if not job:
            return JSONResponse(
                status_code=404,
                content={"status": False, "message": "Job role not found or unauthorized"}
            )

        # 4️⃣ Update job status
        await job_collection.update_one(
            {"_id": ObjectId(body.job_id)},
            {"$set": {"is_active": body.new_status, "updated_at": datetime.utcnow()}}
        )

        return {
            "status": True,
            "message": f"Job status updated to {body.new_status}"
        }

    except Exception as e:
        error_msg = f"Error updating job status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": error_msg}
        )
    
@app.get("/old-job-candidates/{job_id}")
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
                    "video_interview_attended": bool(profile.get("processed_video_url")),
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
class ApplicationSeenByManagerRequest(BaseModel):
    application_id: str
    seen:bool
@app.post("/application-seen-by-manager/")
async def application_seen_by_manager(
    body: ApplicationSeenByManagerRequest,
):
    application= body.application_id
    seen= body.seen
    """
    Mark an application as seen or unseen by the hiring manager.
    """
    try:
        collection = db["resume_profiles"]
        application_doc = await collection.find_one({"_id": ObjectId(application)})
        if not application_doc:
            return JSONResponse(
                content={"status": False, "message": "Application not found"},
                status_code=404
            )
        await collection.update_one(
            {"_id": ObjectId(application)},
            {"$set": {"seen_by_manager": seen, "seen_at": datetime.utcnow()}}
        )
        return JSONResponse(
            content={"status": True, "message": f"Application marked as {'seen' if seen else 'unseen'}"},
            status_code=200 
        )
    except Exception as e:  
        error_msg = f"Error updating application seen status: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": False, "message": error_msg}
        )


class GoogleLoginRequest(BaseModel):
    token: str  # Google ID token from frontend
    role: str   # "candidate", "manager", or "admin"

from auth_utils import verify_google_token_with_library

@app.post("/login-with-google/")
async def login_with_google(payload: GoogleLoginRequest):
    role = payload.role.lower()

    # --- Role → Collection mapping ---
    collection_map = {
        "candidate": "user_accounts",
        "manager": "hiring_managers",
        "admin": "admin_accounts"
    }

    if role not in collection_map:
        raise HTTPException(status_code=400, detail="Invalid role provided")

    collection = db[collection_map[role]]

    # --- Step 1: Verify Google Token using your existing function ---
    try:
        id_info = verify_google_token_with_library(payload.token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")

    # --- Step 2: Extract fields from verified token ---
    google_id = id_info["sub"]
    email = id_info.get("email")
    name = id_info.get("name")
    picture = id_info.get("picture", "")
    email_verified = id_info.get("email_verified", False)

    if not email_verified:
        raise HTTPException(status_code=400, detail="Email not verified by Google")

    # --- Step 3: User handling ---
    user = await collection.find_one({"email": email})
    new_signup = False

    if user:
        user_id = str(user["_id"])
    else:
        new_user = {
            "email": email,
            #"google_id": google_id,
            "name": name,
            "profile_picture": picture,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = await collection.insert_one(new_user)
        user_id = str(result.inserted_id)
        user = new_user
        new_signup = True

    # --- Step 4: Generate tokens ---
    access_token = create_access_token(user_id, role=role)
    refresh_token = create_refresh_token(user_id, role=role)

    await db["refresh_tokens"].delete_many({"user_id": user_id, "user_type": role})
    await save_refresh_token(user_id, user_type=role, refresh_token=refresh_token)

    # --- Step 5: Role-specific responses ---
    if role == "candidate":
        resume_collection = db["resume_profiles"]
        job_collection = db["job_roles"]

        applications_cursor = resume_collection.find({"user_id": user_id})
        applications = await applications_cursor.to_list(length=None)

        job_ids = list({app.get("job_id") for app in applications if app.get("job_id")})
        jobs = []
        if job_ids:
            jobs_cursor = job_collection.find(
                {"_id": {"$in": [ObjectId(jid) for jid in job_ids]}}
            )
            jobs = await jobs_cursor.to_list(length=None)

        job_map = {
            str(job["_id"]): job.get("basicInfo", {}).get("jobTitle", "Unknown") for job in jobs
        }

        application_history = []
        for app in applications:
            job_id = app.get("job_id")
            job_role_name = job_map.get(job_id, "Unknown")
            application_history.append({
                "application_id": str(app["_id"]),
                "job_role_name": job_role_name,
                "job_id": job_id,
                "application_status": app.get("application_status", ""),
                "video_interview_start": app.get("video_interview_start", False),
                "video_email_sent": app.get("video_email_sent", False),
                "audio_interview_status": app.get("audio_interview", False),
            })

        return {
            "status": True,
            "message": "Login successful",
            "new_signup": new_signup,
            "data": {
                "role": role,
                "candidate_id": user_id,
                "name": user.get("name"),
                "email": user.get("email"),
                "profile_picture": user.get("profile_picture", ""),
                "application_history": application_history
            },
            "is_profile_complete":{
                "resume_uploaded": bool(user.get("resume_text")),
                "profile_information": bool(user.get("basic_information")),
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    elif role == "manager":
        return {
            "status": True,
            "message": "Login successful",
            "new_signup": new_signup,
            "data": {
                "role": role,
                "manager_id": user_id,
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "email": user.get("email"),
                "profile_picture": user.get("profile_picture", "")
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

    elif role == "admin":
        return {
            "status": True,
            "message": "Login successful",
            "new_signup": new_signup,
            "data": {
                "role": role,
                "admin_id": user_id,
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "email": user.get("email"),
                "profile_picture": user.get("profile_picture", "")
            },
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }

class UpdateProfileRequest(BaseModel):
    #id: str = Field(..., description="MongoDB ObjectId of the user")
    #role: str = Field(..., description="Role of the user (candidate / hiring_manager / admin)")
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    name: Optional[str] = None
    phone_number: Optional[str] = None
    candidate_source: Optional[str] = None
    phone: Optional[str] = None
    linked_in: Optional[str] = None

@app.post("/update-profile/")
async def update_profile(payload: UpdateProfileRequest, authorization: str = Header(...)):
    try:
        # Validate ObjectId
        try:
            #user_id = ObjectId(payload.id)
            current_user = await get_current_user(authorization)
        except HTTPException as e:
            return JSONResponse(
                content={"status": False, "message": "Invalid or expired token"},
                status_code=401
            )
        user_id=current_user["user_id"]
        logger.info(f"Updating profile for user_id: {user_id}")
        # Determine collection name
        role = current_user["role"]
        logger.info(f"User role: {role}")
        if role == "candidate":
            collection = db["user_accounts"]
        elif role == "manager":
            collection = db["hiring_managers"]
        elif role == "superadmin":
            collection = db["admin_accounts"]
        else:
            raise HTTPException(status_code=400, detail="Invalid role type")

        # Build update dict from non-null fields
        update_fields = {
            k: v for k, v in payload.dict().items()
            if v is not None and k not in ("id", "role")
        }

        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Perform update
        result = await collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_fields})

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "status": True,
            "message": "Profile updated successfully",
            "updated_fields": update_fields
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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
    - If multiple filters are applied, all must be satisfied (AND condition).
    - If no filters are applied, fetch all candidates for that job.
    """
    try:
        job_collection = db["job_roles"]
        job = await job_collection.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job role not found")
        
        profile_collection = db["resume_profiles"]
        user_collection = db["user_accounts"]

        # --- Base condition ---
        conditions = [{"job_id": job_id}]

        # --- Apply filters (each adds an AND condition) ---
        if audio_attended is not None:
            if audio_attended:
                conditions.append({"audio_interview": True})
            else:
                conditions.append({
                    "$or": [
                        {"audio_interview": False},
                        {"audio_interview": {"$exists": False}}
                    ]
                })

        if video_attended is not None:
            if video_attended:
                conditions.append({"video_interview_start": True})
            else:
                conditions.append({
                    "$or": [
                        {"video_interview_start": False},
                        {"video_interview_start": {"$exists": False}}
                    ]
                })

        if video_interview_sent is not None:
            if video_interview_sent:
                conditions.append({"video_email_sent": True})
            else:
                conditions.append({
                    "$or": [
                        {"video_email_sent": False},
                        {"video_email_sent": {"$exists": False}}
                    ]
                })

        if shortlisted is not None:
            if shortlisted:
                conditions.append({"final_shortlist": True})
            else:
                conditions.append({
                    "$or": [
                        {"final_shortlist": False},
                        {"final_shortlist": {"$exists": False}}
                    ]
                })

        if call_for_interview is not None:
            if call_for_interview:
                conditions.append({"call_for_interview": True})
            else:
                conditions.append({
                    "$or": [
                        {"call_for_interview": False},
                        {"call_for_interview": {"$exists": False}}
                    ]
                })

        if application_status is not None:
            if application_status == "SendVideoLink":
                conditions.append({
                    "$or": [
                        {"video_email_sent": True},
                        {"application_status": "SendVideoLink"}
                    ]
                })
            else:
                conditions.append({"application_status": application_status})

        # --- Build final query ---
        has_filters = len(conditions) > 1
        if has_filters:
            filter_conditions = {"$and": conditions}
        else:
            # Default: get all candidates for the given job
            filter_conditions = {"job_id": job_id}

        logger.info(f"Contacts CSV filters: {filter_conditions}")

        # --- Aggregation pipeline ---
        pipeline = [
            {"$match": filter_conditions},
            {"$addFields": {"user_id_object": {"$toObjectId": "$user_id"}}},
            {
                "$lookup": {
                    "from": "user_accounts",
                    "localField": "user_id_object",
                    "foreignField": "_id",
                    "as": "user_info"
                }
            },
            {"$unwind": "$user_info"},
            {
                "$project": {
                    "_id": 1,
                    "name": "$user_info.name",
                    "phone": "$user_info.phone",
                    "email": "$user_info.email"
                }
            }
        ]

        cursor = profile_collection.aggregate(pipeline)

        # --- Generate CSV ---
        csv_output = io.StringIO()
        writer = csv.writer(csv_output)
        writer.writerow(["Name", "Mobile", "Email"])

        async for doc in cursor:
            email = doc.get("email")
            if not email:
                continue

            name = doc.get("name", "N/A")
            phone = doc.get("phone", "N/A")

            if phone and phone != "N/A":
                phone = re.sub(r'^\+91', '', phone)
                phone = re.sub(r'\D', '', phone)
                if len(phone) > 10:
                    phone = phone[-10:]

            writer.writerow([name, phone, email])

        csv_content = csv_output.getvalue()
        csv_output.close()

        filename = f"contacts_{job_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.exception(f"Error retrieving contacts CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving contacts CSV: {str(e)}")
######################################################################
if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)
