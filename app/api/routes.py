from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from app.services.pdf_processor import extract_pages_from_pdf
from app.services.topic_extractor import extract_topics_per_page
from app.services.quiz_generator import generate_quiz_questions
from app.services.chapter_analyzer import analyze_chapters
from typing import Dict, Any, Optional
from pydantic import BaseModel
import os
import shutil
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("uploads")
CHUNKS_DIR = UPLOAD_DIR / "chunks"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)

class CompleteUploadRequest(BaseModel):
    fileId: str
    fileName: str
    totalChunks: int

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
async def complete_upload(request: CompleteUploadRequest) -> Dict[str, Any]:
    """Combine chunks into final file and analyze"""
    chunk_dir = CHUNKS_DIR / request.fileId
    final_path = UPLOAD_DIR / request.fileName
    
    try:
        with open(final_path, "wb") as outfile:
            for i in range(request.totalChunks):
                chunk_path = chunk_dir / f"chunk_{i}"
                with open(chunk_path, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
        
        # Clean up chunks
        shutil.rmtree(chunk_dir)
        
        # Analyze the combined file
        with open(final_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()
            pages = extract_pages_from_pdf(pdf_content)
            topics = extract_topics_per_page(pages)
            chapters = analyze_chapters(pages)
        
        
        return {
            "message": "File upload and analysis completed successfully",
            "chapters": chapters,
            "topics": topics
        }
    except Exception as e:
        # Clean up in case of error
        if final_path.exists():
            os.remove(final_path)
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/generate-quiz")
async def generate_quiz(
    fileName: str = Form(...),
    chapter: str = Form(...),
    page: int = Form(...)
) -> Dict[str, Any]:
    """Generate quiz questions based on the provided PDF content"""
    try:
        
        file_path = UPLOAD_DIR / fileName
        
        # Extract text from the PDF
        with open(file_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()
            pages = extract_pages_from_pdf(pdf_content)
            
            # Get the specific page content if page number is provided
            if page is not None and 0 <= page < len(pages):
                text_content = pages[page]
            else:
                text_content = "\n".join(pages)
            
            # Extract topic from the content
            topics = extract_topics_per_page([text_content])
            chapters = analyze_chapters([text_content])
            topic = topics[0] if topics else "General Knowledge"
            chapter = chapters[0] if chapters else "General Knowledge"
            # Generate quiz questions
            quiz_questions = generate_quiz_questions(text_content, topic, chapter)
            
            # Structure the response
            quiz_response = {
                "quizzes": {
                    f"page_{page if page is not None else 'all'}": {
                        "topic": topic,
                        "chapter": chapter,
                        "page": page if page is not None else 0,
                        "questions": {
                            "multiple_choice": quiz_questions.get("multiple_choice", []),
                            "fill_blanks": quiz_questions.get("fill_blanks", []),
                            "true_false": quiz_questions.get("true_false", []),
                            "matching": quiz_questions.get("matching", [])
                        }
                    }
                }
            }
        
        # Clean up
        os.remove(file_path)
        
        return quiz_response
    except Exception as e:
        # Clean up in case of error
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
