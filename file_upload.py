import anthropic
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from dotenv import dotenv_values
import io

env_vars = dotenv_values(".env")
# client = OpenAI(api_key=env_vars.get("OPENAI_API_KEY"))
client = anthropic.Anthropic(api_key=env_vars.get("ANTROPIC_API_KEY"))

app = FastAPI(title="GpsLaw.AI File Upload API")

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    # file should be pdf or image or document
    
    # if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg", "application/msword",
    #                          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
    if file.content_type not in ["application/pdf", "text/plain", "image/png", "image/jpeg", "image/gif", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only image, PDF, and document files are accepted."
        )
        
    try:
        file_content = await file.read()
            
        file_bytes = io.BytesIO(file_content)
        file_bytes.name = file.filename 
        mime_type = file.content_type
        
        uploaded_file = client.beta.files.upload(
            file=(file.filename, file_bytes, file.content_type)
        )
        # uploaded_file = client.files.create(
        #     file=file_bytes,
        #     purpose="user_data"
        # )
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_id": uploaded_file.id,
            "mime_type": mime_type
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )
        
        


# file-SgyFB58nw9FoCAxC8BKn4j