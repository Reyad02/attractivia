import json
import os
from time import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import dotenv_values

env_vars = dotenv_values(".env")
import re

client = OpenAI(api_key=env_vars.get("OPENAI_API_KEY"))

app = FastAPI(title="GpsLaw.AI Chat API")

SESSIONS_FILE = "chat_sessions.json"


def load_sessions():
    """Load all chat sessions from file."""
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_sessions(sessions):
    """Save all chat sessions to file."""
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def get_or_create_session(sessions, session_id=None):
    """Return existing session if found, otherwise create a new one."""
    if session_id and session_id in sessions:
        return session_id
    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = []
    return new_id

system_prompt = """
You are GpsLaw.AI — a legal guidance engine that behaves like a “GPS of the Law”.
Your goal is to follow a strict sequence: LOCATE -> DIAGNOSE -> GUIDE -> ANTICIPATE.

### OPERATIONAL RULES:
1. ONE QUESTION AT A TIME: You must never ask two questions in one response.
2. STEP-BY-STEP:
   - Phase 1 (Locate): Determine Country, Legal System, and Jurisdiction. 
   - Phase 2 (Diagnose): Ask 1-3 discriminating questions about the case (e.g., dates, contract types).
   - Phase 3 (Guide/Anticipate): Only provide the full structured guidance once Phase 1 and 2 are complete.
3. OUTPUT FORMAT: You must ALWAYS respond in valid JSON.
4. GUIDANCE LOCK: If you are still asking questions (Phase 1 or 2), the "legal_guidance" all object MUST be empty.

### RESPONSE JSON STRUCTURE:
{
    "message": "<Your single question>",
    "localization": {
        "country": "<Country>",
        "legal_system": "<Legal System>",
        "jurisdiction": "<Jurisdiction>",
        "legal_domain": "<Legal Domain>"  
    },
    "legal_guidance":{
        "current_situation": "<A clear statement of who is legally favored>",
        "priority_action": "<One clear action the user should take immediately>",
        "what_to_avoid": [
            "<Common mistake 1>",
            "<Common mistake 2>"
        ],
        "consequences_of_inaction": "<Brief explanation of likely consequences if no action is taken>",
        "anticipation_projection": {
            "next_steps_if_action_fails": "<What happens if the priority action fails>",
            "typical_outcome": "<Typical outcome in similar cases>",
            "estimated_timeline": "<Estimated timeline if possible>"
        },
    }
    "legal_guidance_generation": <True/False> // True if legal_guidance is populated, False if still in questioning phase
}

### PHASE 1: LOCALIZATION (Mandatory)
- If the user's location (Country/State) is unknown, ask: "Which country (and state/province if applicable) is this happening in?"
- Once location is known, explicitly state the Country, Legal System, and Jurisdiction in your "message" and move to Phase 2.

### PHASE 2: DIAGNOSIS
- Ask only discriminating questions (Max 3). 
- Example: "Was your contract permanent (CDI) or fixed-term (CDD)?"

### PHASE 3: GUIDANCE
- Only when you have enough info, populate the "legal_guidance" object using the 4 blocks and the Anticipation section.
"""

TEXT_FORMAT = {
    "type": "json_schema",
    "name": "gpslaw_response",
    "schema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Either a single clarification question or the legal guidance message"
            },
            "localization": {
                "type": "object",
                "properties": {
                    "country": { "type": "string" },
                    "legal_system": { "type": "string" },
                    "jurisdiction": { "type": "string" },
                    "legal_domain": { "type": "string" }
                },
                "required": [
                    "country",
                    "legal_system",
                    "jurisdiction",
                    "legal_domain"
                ],
                "additionalProperties": False
            },
            "legal_guidance": {
                "type": "object",
                "properties": {
                    "current_situation": { "type": "string" },
                    "priority_action": { "type": "string" },
                    "what_to_avoid": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "consequences_of_inaction": { "type": "string" },
                    "anticipation_projection": {
                        "type": "object",
                        "properties": {
                            "next_steps_if_action_fails": { "type": "string" },
                            "typical_outcome": { "type": "string" },
                            "estimated_timeline": { "type": "string" }
                        },
                        "required": [
                            "next_steps_if_action_fails",
                            "typical_outcome",
                            "estimated_timeline"
                        ],
                        "additionalProperties": False
                    }
                },
                "required": [
                    "current_situation",
                    "priority_action",
                    "what_to_avoid",
                    "consequences_of_inaction",
                    "anticipation_projection"
                ],
                "additionalProperties": False
            },
            "legal_guidance_generation": {
                "type": "boolean",
                "description": "False while asking questions, True when legal_guidance is populated"
            }
        },
        "required": [
            "message",
            "localization",
            "legal_guidance",
            "legal_guidance_generation"
        ],
        "additionalProperties": False
    },
    "strict": True
}

class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str


@app.post("/chat")
def chat(request: ChatRequest):
    start_time = time()
    sessions = load_sessions()

    session_id = get_or_create_session(sessions, request.session_id)

    conversation_text = ""
    for m in sessions[session_id]:
        conversation_text += f"User: {m['user_message']}\nAI: {m['ai_message']}\n"
    conversation_text += f"User: {request.user_input}\nAI:"

    try:
        response = client.responses.create(
            # model="gpt-5", -> question asked 4
            model="gpt-5.1", 
            # -> question asked 6
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            text={
                "format": TEXT_FORMAT
            }
        )

        end_time = time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(response.output_text)

        ai_reply = response.output_text

        # Remove markdown code blocks if present
        ai_reply = re.sub(r'^```json\s*', '', ai_reply)
        ai_reply = re.sub(r'^```\s*', '', ai_reply)
        ai_reply = re.sub(r'\s*```$', '', ai_reply)
        ai_reply = ai_reply.strip()
        
        response_json = json.loads(ai_reply)
        ai_message = response_json.get("message", "")
        legal_guidance = response_json.get("legal_guidance", {})
        localization = response_json.get("localization", {})
        legal_guidance_generation = response_json.get("legal_guidance_generation", False)
        
        conversation_entry = {
            "user_message": request.user_input,
            "ai_message": ai_message,
        }
        
        if legal_guidance:
            conversation_entry["legal_guidance"] = legal_guidance
        if localization:
            conversation_entry["localization"] = localization

        # check the legal_guidance_generation flag. based on that, we can decide whether full guidance is generate or not 
        
        sessions[session_id].append(conversation_entry)
        save_sessions(sessions)

        return {
            "session_id": session_id,
            "response": {
                "message": ai_message,
                "localization": localization,
                "legal_guidance": legal_guidance,
                "legal_guidance_generation": legal_guidance_generation
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
