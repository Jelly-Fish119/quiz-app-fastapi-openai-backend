from fastapi import APIRouter, UploadFile, File
from app.services.topic_extractor import extract_topics
from app.services.quiz_generator import generate_quiz_questions
from app.services.chapter_analyzer import analyze_chapters
from app.services.pdf_processor import process_pdf
from typing import Dict, Any

router = APIRouter()

@router.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze a PDF file to extract chapters, topics, and their positions.
    """
    try:
        # Process the PDF and get text content
        text_content = await process_pdf(file)
        
        # Analyze chapters and topics
        analysis = analyze_chapters(text_content)
        
        return analysis
    except Exception as e:
        return {"error": str(e)}

# ... (keep existing endpoints) 