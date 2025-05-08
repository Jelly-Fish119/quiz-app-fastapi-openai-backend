from PyPDF2 import PdfReader
from typing import List
from io import BytesIO

def extract_pages_from_pdf(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    return [page.extract_text() for page in reader.pages if page.extract_text()]
