from fastapi import APIRouter, File, UploadFile
from app.services.pdf_processor import extract_pages_from_pdf
from app.services.topic_extractor import extract_topics_per_page

router = APIRouter()

@router.post("/extract-topics/")
async def extract_topics(pdf_file: UploadFile = File(...)):
    content = await pdf_file.read()
    pages = extract_pages_from_pdf(content)
    topics = extract_topics_per_page(pages)
    return {"topics": topics}
