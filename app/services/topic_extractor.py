from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
import logging

genai.configure(api_key=GOOGLE_API_KEY)

def extract_topic_from_text(text: str) -> str:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Extract the main topic from the following text. Return only the topic, nothing else.
        
        Text:
        {text}
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error in topic extraction: {str(e)}")
        return "Topic extraction failed"

def extract_topics_per_page(pages: list[str]) -> list[str]:
    return [extract_topic_from_text(page) for page in pages]
