from fastapi import APIRouter, File, UploadFile
from app.services.pdf_processor import extract_pages_from_pdf
from app.services.topic_extractor import extract_topics_per_page
from app.services.quiz_generator import generate_quiz_questions
from app.services.chapter_analyzer import analyze_chapters
from typing import Dict, Any

router = APIRouter()

@router.post("/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and process a PDF file to extract topics and generate questions.
    """
    try:
        content = await file.read()
        pages = extract_pages_from_pdf(content)
        topics = extract_topics_per_page(pages)
        
        # Generate quiz questions for each page
        quizzes = {}
        for i, (page, topic) in enumerate(zip(pages, topics), 1):
            questions = generate_quiz_questions(page, topic)
            quizzes[f"page_{i}"] = {
                "topic": topic,
                "questions": questions
            }
        
        return {"quizzes": quizzes}
    except Exception as e:
        return {"error": str(e)}

@router.post("/pdf/analyze")
async def analyze_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze a PDF file to extract chapters, topics, and their positions.
    """
    try:
        content = await file.read()
        pages = extract_pages_from_pdf(content)
        text_content = {f"page_{i}": page for i, page in enumerate(pages, 1)}
        analysis = analyze_chapters(text_content)
        return analysis
    except Exception as e:
        return {"error": str(e)}
