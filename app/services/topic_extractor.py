from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
import logging

genai.configure(api_key=GOOGLE_API_KEY)

def extract_topic_from_text(text: str) -> str:
    try:
        # List available models
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name
                break
        else:
            raise ValueError("No suitable model found for content generation")

        model = genai.GenerativeModel(model_name)
        
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
