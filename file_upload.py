from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from dotenv import dotenv_values
import io

env_vars = dotenv_values(".env")
client = OpenAI(api_key=env_vars.get("OPENAI_API_KEY"))

app = FastAPI(title="GpsLaw.AI File Upload API")

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF files are accepted."
        )
        
    try:
        file_content = await file.read()
            
        file_bytes = io.BytesIO(file_content)
        file_bytes.name = file.filename 
        
        uploaded_file = client.files.create(
            file=file_bytes,
            purpose="user_data"
        )
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_id": uploaded_file.id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )
        
        


# file-SgyFB58nw9FoCAxC8BKn4j