from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from app.services.pdf_processor import extract_pages_from_pdf
from app.services.topic_extractor import extract_topics_per_page
from app.services.quiz_generator import generate_quiz_questions
from app.services.chapter_analyzer import analyze_chapters
from typing import Dict, Any, Optional

router = APIRouter()

@router.post("/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a PDF file and generate quiz questions for each page.
    """
    try:
        # Read the PDF file
        pdf_bytes = await file.read()
        
        # Extract pages from PDF
        pages = extract_pages_from_pdf(pdf_bytes)
        
        # Extract topics for each page
        topics_per_page = extract_topics_per_page(pages)
        
        # Generate quiz questions for each page
        quizzes = {}
        for page_num, topics in topics_per_page.items():
            page_content = pages[page_num - 1]  # Convert to 0-based index
            quiz = generate_quiz_questions(page_content, topics)
            quizzes[f"page_{page_num}"] = {
                "topic": ", ".join(topics),
                "questions": quiz
            }
        
        return {"quizzes": quizzes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/analyze")
async def analyze_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze a PDF file to extract chapters and their topics.
    """
    try:
        # Read the PDF file
        pdf_bytes = await file.read()
        
        # Extract pages from PDF
        pages = extract_pages_from_pdf(pdf_bytes)
        
        # Analyze chapters and their topics
        chapters = analyze_chapters(pages)
        
        return {"chapters": chapters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    chapter: Optional[str] = Query(None, description="Chapter title to generate quiz for"),
    page: Optional[int] = Query(None, description="Page number to generate quiz for")
) -> Dict[str, Any]:
    """
    Generate quiz questions for a specific chapter or page.
    """
    try:
        # Read the PDF file
        pdf_bytes = await file.read()
        
        # Extract pages from PDF
        pages = extract_pages_from_pdf(pdf_bytes)
        
        # Analyze chapters and their topics
        chapters = analyze_chapters(pages)
        
        # Filter pages based on chapter or page selection
        selected_pages = []
        selected_topics = []
        
        if chapter:
            # Find the chapter and its pages
            for ch in chapters:
                if ch["title"].lower() == chapter.lower():
                    selected_pages = pages[ch["pageNumber"]-1:ch["pageNumber"]]
                    selected_topics = [topic["name"] for topic in ch["topics"]]
                    break
        elif page is not None:
            # Select specific page
            if 1 <= page <= len(pages):
                selected_pages = [pages[page-1]]
                # Extract topics for the selected page
                topics_per_page = extract_topics_per_page([pages[page-1]])
                selected_topics = topics_per_page[0] if topics_per_page else []
        
        if not selected_pages:
            raise HTTPException(status_code=400, detail="No valid chapter or page selected")
        
        # Generate quiz questions for selected pages
        quizzes = {}
        for i, page_content in enumerate(selected_pages):
            page_num = page if page is not None else chapters[i]["pageNumber"]
            quiz = generate_quiz_questions(page_content, selected_topics)
            quizzes[f"page_{page_num}"] = {
                "topic": ", ".join(selected_topics),
                "chapter": chapter if chapter else None,
                "page": page_num,
                "questions": quiz
            }
        
        return {"quizzes": quizzes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
