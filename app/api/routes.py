from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from app.services.pdf_processor import extract_pages_from_pdf
from app.services.topic_extractor import extract_topics_per_page, extract_topics
from app.services.quiz_generator import generate_quiz_questions
from app.services.chapter_analyzer import analyze_chapters
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class Page(BaseModel):
    page_number: int
    text: str

class PageAnalysisRequest(BaseModel):
    pages: List[Page]

class QuizGenerationRequest(BaseModel):
    pageNumber: int
    text: str
    topic: str
    chapter: str

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
            
            if not pages:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract text from PDF. The file might be empty or corrupted."
                )
            
            # Extract topics for all pages in a single call
            topics = extract_topics_per_page(pages)
            
            # Create a topics dictionary with page numbers
            topics_dict = {}
            for i in range(len(pages)):
                if i in topics:
                    topics_dict[i] = topics[i]
                else:
                    topics_dict[i] = "General Knowledge"
            
            # Analyze chapters
            chapters = analyze_chapters(pages)
        
        return {
            "message": "File upload and analysis completed successfully",
            "chapters": chapters,
            "topics": topics_dict,
            "fileName": request.fileName
        }
    except Exception as e:
        # Clean up in case of error
        if final_path.exists():
            os.remove(final_path)
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/analyze-pages")
async def analyze_pages(request: PageAnalysisRequest):
    try:
        # Extract topics for all pages
        topics = await extract_topics_per_page([page.text for page in request.pages])
        
        # Analyze chapters
        chapters = await analyze_chapters([page.text for page in request.pages])
        
        # Generate quiz questions
        questions = await generate_quiz_questions(
            [page.text for page in request.pages],
            topics,
            chapters
        )
        
        return {
            "topics": topics,
            "chapters": chapters,
            "questions": questions
        }
    except Exception as e:
        logger.error(f"Error analyzing pages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while analyzing pages: {str(e)}")

@router.post("/api/analyze-page")
async def analyze_page(request: PageAnalysisRequest) -> Dict[str, Any]:
    """Analyze a single page for topics and chapters"""
    try:
        # Extract topics
        topics = extract_topics([request.text])
        
        # Analyze chapters
        chapters = analyze_chapters([request.text])
        
        return {
            "topics": topics,
            "chapters": chapters
        }
    except Exception as e:
        logger.error(f"Error analyzing page: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze page: {str(e)}"
        )

@router.post("/api/generate-quiz")
async def generate_quiz(request: QuizGenerationRequest) -> Dict[str, Any]:
    """Generate quiz questions for a specific page"""
    try:
        # Generate quiz questions
        quiz_questions = generate_quiz_questions(
            request.text,
            request.topic,
            request.chapter
        )
        
        return {
            "topic": request.topic,
            "chapter": request.chapter,
            "questions": quiz_questions
        }
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate quiz: {str(e)}"
        )