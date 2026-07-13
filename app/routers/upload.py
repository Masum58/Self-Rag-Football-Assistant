import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.ingestion.build_index import process_single_file, SAMPLE_DOCS_DIR
from app.vectorstore.pinecone_client import upsert_documents
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a file from the user, saves it permanently to the sample_docs directory,
    processes it into chunks, and uploads the chunks to Pinecone.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Ensure the upload directory exists
    SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    file_path = SAMPLE_DOCS_DIR / file.filename

    # Save the file permanently
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    finally:
        file.file.close()

    # Process and upload to Pinecone
    try:
        chunks = process_single_file(file_path)
        if not chunks:
            raise ValueError("No text could be extracted or unsupported file type.")
            
        upsert_documents(chunks)
        logger.info(f"Successfully processed and uploaded {len(chunks)} chunks for {file.filename}")
        
        return UploadResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"File successfully uploaded and {len(chunks)} chunks ingested."
        )
    except Exception as e:
        logger.error(f"Failed to process and upload chunks: {e}")
        raise HTTPException(status_code=500, detail=f"File saved, but ingestion failed: {str(e)}")
