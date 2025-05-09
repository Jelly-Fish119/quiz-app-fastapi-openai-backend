from fastapi import APIRouter, File, UploadFile
from app.services.pdf_processor import extract_pages_from_pdf
from app.services.topic_extractor import extract_topics_per_page
from app.services.quiz_generator import generate_quiz_questions

router = APIRouter()

@router.post("/extract-topics/")
async def extract_topics(pdf_file: UploadFile = File(...)):
    content = await pdf_file.read()
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
