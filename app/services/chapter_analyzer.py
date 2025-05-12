from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
import logging

# Configure logging
logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

def analyze_chapters(texts: list[str]) -> list[str]:
    """
    Analyze text segments to identify chapters using Google's Generative AI.
    Returns a list of chapter names.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Analyze the following text and identify the chapter or section it belongs to.
        Text:
        {texts[0]}

        Return ONLY a list of chapter names, one per line.
        Each chapter name should be a short phrase (2-5 words).
        Do not include any explanations or additional text.
        """
        
        response = model.generate_content(prompt)
        chapters = [chapter.strip() for chapter in response.text.strip().split('\n') if chapter.strip()]
        
        return chapters
    except Exception as e:
        logger.error(f"Error analyzing chapters: {str(e)}")
        return ["General Content"] 