import aiohttp
from fastapi import HTTPException
from dotenv import load_dotenv
import os

load_dotenv()
class Settings:
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "  ")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", " ")
    AZURE_DEPLOYMENT_NAME: str = os.getenv("AZURE_DEPLOYMENT_NAME", " ")
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", " ")


settings = Settings()
# Headers for Azure OpenAI API
AZURE_HEADERS = {
    "Content-Type": "application/json",
    "api-key": settings.AZURE_OPENAI_KEY
}
AZURE_OPENAI_URL = f"{settings.AZURE_OPENAI_ENDPOINT}/openai/deployments/{settings.AZURE_DEPLOYMENT_NAME}/chat/completions?api-version={settings.AZURE_API_VERSION}"
async def call_openai_for_jd(job_data: dict) -> str:
    prompt = f"""
    You are an expert recruiter tasked with generating a job description based on the provided structured data.
    Here is the job data:
    {job_data}
    Generate a concise and engaging job description that highlights the key responsibilities, qualifications, and benefits of the role. The description should be suitable for posting on job boards and attracting qualified candidates.
    Make sure to include all necessary details in the job description.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that generates job descriptions based on structured data."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 2000
            }
        ) as response:
            if response.status != 200:
                detail = await response.text()
                raise HTTPException(status_code=500, detail=f"Error calling LLM: {detail}")

            result = await response.json()
            try:
                return result["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                raise HTTPException(status_code=500, detail="Invalid response format from LLM")