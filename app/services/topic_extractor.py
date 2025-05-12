from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
import logging

# Configure logging
logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

def extract_topics(texts: list[str]) -> list[str]:
    """
    Extract topics from a list of text segments using Google's Generative AI.
    Returns a list of topics.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Analyze the following text and extract the main topics.
        Text:
        {texts[0]}

        Return ONLY a list of topics, one per line.
        Each topic should be a short phrase (2-5 words).
        Do not include any explanations or additional text.
        """
        
        response = model.generate_content(prompt)
        topics = [topic.strip() for topic in response.text.strip().split('\n') if topic.strip()]
        
        return topics
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        return ["General Knowledge"]