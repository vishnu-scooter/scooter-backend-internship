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
async def call_openai_for_jd(jd_text: str) -> dict:
    """
    Parse an arbitrary job description text and return structured job data
    in the same format as JDCreationBody.
    Fields that cannot be inferred are returned as None / null.
    """
    prompt = f"""
    You are an expert recruiter .
    assume and Extract structured job information from the following job description:

    {jd_text}

    Return the output in JSON format exactly matching this structure:

    {{
        "basicInfo": {{
            "companyName": string or null,
            "jobTitle": string or null,
            "roleType": string or null,
            "primaryFocus": [list of strings] or null,
            "salesProcessStages": [list of strings] or null
        }},
        "experienceSkills": {{
            "minExp": number or null,
            "maxExp": number or null,
            "skillsRequired": [list of strings] or null,
            "workLocation": "inPerson" | "hybrid" | "remote" or null,
            "location": [list of strings] or null,
            "timeZone": [list of strings] or null
        }},
        "compensations": {{
            "baseSalary": {{
                "currency": string or null,
                "minSalary": number or null,
                "maxSalary": number or null,
                "cadence": string or null
            }} or null,
            "ote": [list of strings] or null,
            "equityOffered": true/false or null,
            "opportunities": [list of strings] or null,
            "keyChallenged": [list of strings] or null,
            "laguages": [list of strings] or null
        }} or null
    }}

    For any field that cannot be determined from the text, return null.
    ONLY return valid JSON. Do not add extra explanations.
    """

    async with aiohttp.ClientSession() as session:
        async with session.post(
            AZURE_OPENAI_URL,
            headers=AZURE_HEADERS,
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that extracts structured job info from text."},
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
                jd_json_str = result["choices"][0]["message"]["content"].strip()
                # Convert JSON string to dict
                import json
                return json.loads(jd_json_str)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                raise HTTPException(status_code=500, detail=f"Invalid response format from LLM: {str(e)}")