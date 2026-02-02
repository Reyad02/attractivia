import anthropic
from fastapi import FastAPI, HTTPException
from dotenv import dotenv_values
import json
import re

env_vars = dotenv_values(".env")
client = anthropic.Anthropic(api_key=env_vars.get("ANTROPIC_API_KEY"))

app = FastAPI(title="GpsLaw.AI OCR API")

OCR_TEXT_FORMAT = {
    "type": "json_schema",
    # "name": "gpslaw_ocr_response",
    "schema": {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "Indicates whether OCR extraction was successful"
            },
            "data": {
                "type": "string",
                "description": "Extracted raw text from the document"
            }
        },
        "required": [
            "success",
            "data"
        ],
        "additionalProperties": False
    },
    # "strict": True
}

def extract_json(text: str):
    """
    Extract valid JSON from model output.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

@app.post("/extract_user_details")
async def extract_user_details(file_id: str, mime_type: str):
    """
    file_id: OpenAI uploaded file ID (from another API)
    """
    try:
        prompt = """
        You are an OCR data extraction assistant.

        Extract text from the provided document and return ONLY valid JSON.
        No markdown, no explanations, no extra text, no summary.
        """
        
        file_type = "image" if mime_type.startswith("image/") else "document"

        response = client.beta.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": file_type,
                            "source": {
                                "type": "file",
                                "file_id": file_id
                            }
                        }
                    ]
                }
            ],
            output_config={
                "format": OCR_TEXT_FORMAT
            }
            ,
            betas=["files-api-2025-04-14"]
        )

        ai_reply = response.content[0].text
        ai_reply = re.sub(r'^```json\s*', '', ai_reply)
        ai_reply = re.sub(r'^```\s*', '', ai_reply)
        ai_reply = re.sub(r'\s*```$', '', ai_reply)
        ai_reply = ai_reply.strip()

        response_json = json.loads(ai_reply)
        data = response_json.get("data", "")
        success = response_json.get("success", {})

        return {
            "response": {
                "success": success,
                "data": data,
                "mime_type": mime_type
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )
