from fastapi import FastAPI, HTTPException
from openai import OpenAI
from dotenv import dotenv_values
import json
import re

env_vars = dotenv_values(".env")
client = OpenAI(api_key=env_vars.get("OPENAI_API_KEY"))

app = FastAPI(title="GpsLaw.AI DOC Analysis API")

system_prompt = """
Role:
You are an elite AI Legal Counsel and Contract Specialist. Your purpose is to conduct deep-dive analyses of legal documents (contracts, NDAs, Terms of Service, employment agreements, etc.) to protect the user's interests. You possess a meticulous eye for detail, a deep understanding of standard commercial law, and the ability to spot "red flag" clauses that others might miss.

Objective:
Analyze the provided legal document and provide a structured report consisting of:
Executive Summary: A high-level overview of the document’s purpose, the parties involved, and the overall "friendliness" of the agreement toward the user.
Key Clauses & Obligations: A breakdown of the most significant provisions (e.g., payment terms, duration, termination rights, confidentiality).
Potential Risks & Red Flags: Identification of unfavorable terms, ambiguous language, lopsided liabilities, or hidden traps.
Actionable Recommendations: Specific advice on what to negotiate, what to delete, or what to clarify with the counterparty.

Guidelines for Analysis:
Tone: Professional, objective, and authoritative.
Precision: Reference specific sections or paragraph numbers from the text whenever possible.
Critical Thinking: Look for what is missing (e.g., if a contract allows one party to terminate for convenience but not the other).
Clarity: Translate complex "legalese" into plain, actionable English.

Response Structure:
Legal Analysis Report
1. Executive Summary
[Provide a 3-5 sentence summary of the document's intent and overall risk profile.]
2. Key Clauses
[Clause Name]: [Explanation of the obligation or right.]
[Clause Name]: [Explanation of the obligation or right.]
3. Risk Assessment & Red Flags
High Risk: [Detail a specific high-risk clause and why it is dangerous.]
Medium Risk: [Detail a concern that requires attention.]
Ambiguity: [Point out vague language that could lead to disputes.]
4. Strategic Recommendations
Negotiation Point: [What the user should ask for.]
Clarification Needed: [Questions the user should ask the other party.]
Suggested Edit: [How to reword a specific sentence to be safer.]

"""

DOC_TEXT_FORMAT = {
    "type": "json_schema",
    "name": "gpslaw_doc_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "localization": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Country inferred from the documents or 'Unknown'"
                    },
                    "legal_system": {
                        "type": "string",
                        "description": "Legal system (e.g., Common Law, Civil Law, Hybrid) or 'Unknown'"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Specific jurisdiction or authority or 'Unknown'"
                    },
                    "legal_domain": {
                        "type": "string",
                        "description": "Primary legal domain (e.g., Contract Law, Employment Law)"
                    }
                },
                "required": [
                    "country",
                    "legal_system",
                    "jurisdiction",
                    "legal_domain"
                ],
                "additionalProperties": False
            },
            "potential_risks": {
                "type": "array",
                "description": "Identified legal risks based on the documents",
                "items": {
                    "type": "string"
                }
            },
            "key_clauses": {
                "type": "array",
                "description": "Legally significant clauses",
                "items": {
                    "type": "string"
                }
            },
            "ai_recommendation": {
                "type": "array",
                "description": "Legally grounded actions to mitigate risk or improve compliance",
                "items": {
                    "type": "string"
                }
            },
            "summary": {
                "type": "string",
                "description": "Concise, lawyer-style summary of the legal situation and exposure"
            }
        },
        "required": [
            "localization",
            "potential_risks",
            "key_clauses",
            "ai_recommendation",
            "summary"
        ],
        "additionalProperties": False
    },
    "strict": True
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

@app.post("/doc_analysis")
async def extract_user_details(file_id: str):
    """
    file_id: OpenAI uploaded file ID (from another API)
    """
    try:
        prompt = """
        You are an OCR data extraction assistant.

        Extract text from the provided document and return ONLY valid JSON.
        No markdown, no explanations, no extra text, no summary.
        """

        response = client.responses.create(
            model="gpt-5.2",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file_id
                        },
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ],
            text={
                "format": DOC_TEXT_FORMAT
            }
        )

        ai_reply = response.output_text
        ai_reply = re.sub(r'^```json\s*', '', ai_reply)
        ai_reply = re.sub(r'^```\s*', '', ai_reply)
        ai_reply = re.sub(r'\s*```$', '', ai_reply)
        ai_reply = ai_reply.strip()
        
        response_json = json.loads(ai_reply)
        localization = response_json.get("localization", {})
        potential_risks = response_json.get("potential_risks", [])
        key_clauses = response_json.get("key_clauses", [])
        ai_recommendation = response_json.get("ai_recommendation", [])
        summary = response_json.get("summary", "")

        return {
            "response": {
                "summary": summary,
                "localization": localization,
                "potential_risks": potential_risks,
                "key_clauses": key_clauses,
                "ai_recommendation": ai_recommendation
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )
