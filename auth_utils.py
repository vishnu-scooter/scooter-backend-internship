import os, jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import requests
from google.auth.transport import requests
from google.oauth2 import id_token
from fastapi import Header, HTTPException
load_dotenv()

ACCESS_SECRET = os.getenv("JWT_ACCESS_SECRET")
REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_EXPIRE_MINUTES", 60))
REFRESH_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", 7))
YOUR_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL")
client = AsyncIOMotorClient(MONGODB_URL)
db = client["scooter_ai_db"]

class RefreshTokenSchema(BaseModel):
    user_id: str            # str(ObjectId) of company/manager
    user_type: str          # "company" or "manager"
    refresh_token: str
    created_at: datetime

def create_access_token(user_id: str, role: str):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire
    }
    return jwt.encode(payload, ACCESS_SECRET, algorithm=ALGORITHM)

def create_refresh_token(user_id: str, role: str = "manager"):
    expire = datetime.utcnow() + timedelta(days=REFRESH_EXPIRE_DAYS)  # long-lived
    payload = {
        "sub": user_id,
        "role": role,
        "exp": expire
    }
    return jwt.encode(payload, REFRESH_SECRET, algorithm=ALGORITHM)

def verify_access_token(token: str):
    return jwt.decode(token, ACCESS_SECRET, algorithms=[ALGORITHM])

def verify_refresh_token(token: str):
    return jwt.decode(token, REFRESH_SECRET, algorithms=[ALGORITHM])

async def save_refresh_token(user_id: str, user_type: str, refresh_token: str):
    collection = db["refresh_tokens"]
    token_doc = {
        "user_id": user_id,
        "user_type": user_type,
        "refresh_token": refresh_token,
        "created_at": datetime.utcnow()
    }
    await collection.insert_one(token_doc)

async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]

    try:
        payload = verify_access_token(token)
        user_id = payload.get("sub")  # ✅ use "sub", not "user_id"
        role = payload.get("role")
        if not user_id or not role:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return {"user_id": user_id, "role": role}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or tampered token")

    

def verify_google_token_with_library(id_token_string):
    """
    Verify Google ID token using Google's official library
    """
    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(
            id_token_string, 
            requests.Request(), 
            YOUR_GOOGLE_CLIENT_ID
        )
        return idinfo
        
    except ValueError as e:
        return False, f"Invalid token: {str(e)}"
    
