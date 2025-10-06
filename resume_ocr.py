from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
import numpy as np
from fastapi import UploadFile, HTTPException
import io
from dotenv import load_dotenv
import os
load_dotenv()

endpoint = os.getenv("OCR_ENDPOINT")
key = os.getenv("OCR_KEY")

def format_bounding_box(bounding_box):
    if not bounding_box:
        return "N/A"
    reshaped_bounding_box = np.array(bounding_box).reshape(-1, 2)
    return ", ".join([f"[{x}, {y}]" for x, y in reshaped_bounding_box])

async def extract_text_with_ocr(pdf_file: UploadFile) -> str:
    try:
        # Read file content
        content = await pdf_file.read()

        # Create client
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        # Call Azure OCR
        poller = document_intelligence_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=io.BytesIO(content),
            content_type="application/pdf"
        )
        result = poller.result()

        # Page limit check (Azure returns pages too)
        if len(result.pages) > 10:
            raise HTTPException(status_code=400, detail="Resume exceeds typical page limit")

        # Extract text per page
        text = result.content
        # for page_num, page in enumerate(result.pages, 1):
        #     page_text = page.content if page.content else ""
        #     text += page_text + "\n"
        return text.strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
