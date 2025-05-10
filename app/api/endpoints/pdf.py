from fastapi import APIRouter, UploadFile, File
from app.services.topic_extractor import extract_topics
from app.services.quiz_generator import generate_quiz_questions
from app.services.chapter_analyzer import analyze_chapters
from app.services.pdf_processor import process_pdf
from typing import Dict, Any

router = APIRouter()

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and process a PDF file to extract topics and generate questions.
    """
    try:
        # Process the PDF and get text content
        text_content = await process_pdf(file)
        
        # Extract topics
        topics = extract_topics(text_content)
        
        # Generate questions for each topic
        quizzes = {}
        for page_num, page_topics in topics.items():
            page_text = text_content.get(page_num, "")
            quizzes[page_num] = {
                "topic": page_topics[0] if page_topics else "General",
                "questions": generate_quiz_questions(page_text, page_topics[0] if page_topics else "General")
            }
        
        return {"quizzes": quizzes}
    except Exception as e:
        return {"error": str(e)}

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