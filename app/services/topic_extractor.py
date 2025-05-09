from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

def extract_topic_from_text(text: str) -> str:
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""Extract the main topic from the following text. Return only the topic, nothing else.
    
    Text:
    {text}
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()

def extract_topics_per_page(pages: list[str]) -> list[str]:
    return [extract_topic_from_text(page) for page in pages]
