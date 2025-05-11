from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
import logging

genai.configure(api_key=GOOGLE_API_KEY)

def extract_topic_from_text(text: str) -> str:
    print("------------quiz topic extraction------------")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Extract the main topic from the following text. Return only the topic as a single string, nothing else. Do not split the topic into individual characters.
        
        Text:
        {text}
        """
        
        response = model.generate_content(prompt)
        # Ensure we get a proper string and remove any potential character splitting
        topic = response.text.strip()
        # If the topic is still split into characters (comma-separated), join them
        if ',' in topic:
            topic = ''.join(topic.split(','))
        return topic
    except Exception as e:
        logging.error(f"Error in topic extraction: {str(e)}")
        return "Topic extraction failed"

def extract_topics_per_page(pages: list[str]) -> list[str]:
    return [extract_topic_from_text(page) for page in pages]