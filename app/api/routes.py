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

@router.post("/pdf/generate-quiz")
async def generate_quiz(
    fileName: str = Form(...),
    page: Optional[int] = Query(None)
) -> Dict[str, Any]:
    """Generate quiz questions based on the provided PDF content"""
    file_path = UPLOAD_DIR / fileName
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {fileName} not found. Please upload the file first."
        )
    
    try:
        # Extract text from the PDF
        with open(file_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()
            pages = extract_pages_from_pdf(pdf_content)
            
            if not pages:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract text from PDF. The file might be empty or corrupted."
                )
            
            # Get the specific page content if page number is provided
            if page is not None:
                if page < 0 or page >= len(pages):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid page number. The PDF has {len(pages)} pages."
                    )
                text_content = pages[page]
            else:
                text_content = "\n".join(pages)
            
            if not text_content.strip():
                raise HTTPException(
                    status_code=400,
                    detail="No text content found in the selected page(s)."
                )
            
            # Extract topics for all pages in a single call
            try:
                topics = extract_topics_per_page(pages)
                logger.info(f"Extracted topics: {topics}")
            except Exception as e:
                logger.error(f"Error extracting topics: {str(e)}")
                topics = {}
            
            try:
                chapters = analyze_chapters(pages)
                logger.info(f"Extracted chapters: {chapters}")
            except Exception as e:
                logger.error(f"Error analyzing chapters: {str(e)}")
                chapters = {}
            
            # Get the topic for the current page or use the first available topic
            try:
                if page is not None and page in topics:
                    topic = topics[page]
                elif topics:
                    topic = next(iter(topics.values()))
                else:
                    topic = "General Knowledge"
                logger.info(f"Selected topic: {topic}")
            except Exception as e:
                logger.error(f"Error selecting topic: {str(e)}")
                topic = "General Knowledge"
            
            # Get the chapter for the current page or use the first available chapter
            try:
                if page is not None and page in chapters:
                    chapter = chapters[page]
                elif chapters:
                    chapter = next(iter(chapters.values()))
                else:
                    chapter = "General Knowledge"
                logger.info(f"Selected chapter: {chapter}")
            except Exception as e:
                logger.error(f"Error selecting chapter: {str(e)}")
                chapter = "General Knowledge"
            
            # Generate quiz questions
            try:
                quiz_questions = generate_quiz_questions(text_content, topic, chapter)
            except Exception as e:
                logger.error(f"Error generating quiz questions: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate quiz questions: {str(e)}"
                )
            
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
        
        return quiz_response
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        print(f"Error in generate_quiz: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the quiz: {str(e)}"
        )