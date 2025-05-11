from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from app.services.pdf_processor import extract_pages_from_pdf
from app.services.topic_extractor import extract_topics_per_page
from app.services.quiz_generator import generate_quiz_questions
from app.services.chapter_analyzer import analyze_chapters
from typing import Dict, Any, Optional
import os
import shutil
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("uploads")
CHUNKS_DIR = UPLOAD_DIR / "chunks"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)

@router.post("/pdf/upload-chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    fileId: str = Form(...),
    chunkIndex: str = Form(...),
    totalChunks: str = Form(...),
    fileName: str = Form(...)
) -> Dict[str, Any]:
    """Handle individual chunk uploads"""
    chunk_dir = CHUNKS_DIR / fileId
    chunk_dir.mkdir(exist_ok=True)
    
    chunk_path = chunk_dir / f"chunk_{chunkIndex}"
    
    try:
        with open(chunk_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Chunk {chunkIndex} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/complete-upload")
async def complete_upload(
    fileId: str,
    fileName: str,
    totalChunks: int
) -> Dict[str, Any]:
    """Combine chunks into final file"""
    chunk_dir = CHUNKS_DIR / fileId
    final_path = UPLOAD_DIR / fileName
    
    try:
        with open(final_path, "wb") as outfile:
            for i in range(totalChunks):
                chunk_path = chunk_dir / f"chunk_{i}"
                with open(chunk_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
        
        # Clean up chunks
        shutil.rmtree(chunk_dir)
        
        return {"message": "File upload completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/analyze")
async def analyze_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyze PDF structure"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract pages and analyze
        pages = extract_pages_from_pdf(str(file_path))
        topics = extract_topics_per_page(pages)
        chapters = analyze_chapters(pages)
        
        # Clean up
        os.remove(file_path)
        
        return {
            "chapters": chapters,
            "topics": topics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    chapter: Optional[str] = None,
    page: Optional[int] = None
) -> Dict[str, Any]:
    """Generate quiz questions"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate quiz
        quiz = generate_quiz_questions(str(file_path), chapter, page)
        
        # Clean up
        os.remove(file_path)
        
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
