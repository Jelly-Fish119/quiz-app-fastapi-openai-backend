from fastapi import APIRouter, HTTPException
from app.services.topic_extractor import extract_topics
from app.services.chapter_analyzer import analyze_chapters
from app.services.quiz_generator import generate_quiz_questions
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

class Page(BaseModel):
    page_number: int
    text: str

class PageAnalysisRequest(BaseModel):
    pages: List[Page]

class QuizGenerationRequest(BaseModel):
    page_number: int
    text: str
    topic: str
    chapter: str

@router.post("/pdf/analyze-pages")
async def analyze_pages(request: PageAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze multiple pages to extract topics and chapters.
    """
    try:
        results = []
        for page in request.pages:
            # Extract topics
            topics = await extract_topics(page.text)
            
            # Analyze chapters
            chapters = await analyze_chapters(page.text)
            
            results.append({
                "page_number": page.page_number,
                "topics": topics,
                "chapters": chapters
            })
        
        return {
            "pages": results
        }
    except Exception as e:
        logger.error(f"Error analyzing pages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing pages: {str(e)}"
        )

@router.post("/pdf/generate-quiz")
async def generate_quiz(request: QuizGenerationRequest) -> Dict[str, Any]:
    """
    Generate quiz questions for a specific page.
    """
    try:
        # Generate quiz questions
        questions = await generate_quiz_questions(
            request.text,
            request.topic,
            request.chapter
        )
        
        return {
            "page_number": request.page_number,
            "topic": request.topic,
            "chapter": request.chapter,
            "questions": questions
        }
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating quiz: {str(e)}"
        ) 
