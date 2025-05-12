from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
import logging

genai.configure(api_key=GOOGLE_API_KEY)

def extract_topics_per_page(pages: list[str]) -> dict[int, str]:
    """
    Extract topics from all pages using a single Gemini call.
    Returns a dictionary mapping page numbers to their topics.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Combine all pages into a single text with page markers
        combined_text = "\n\n".join([f"Page {i+1}:\n{page}" for i, page in enumerate(pages)])
        print("combined_text\n", combined_text)        
        prompt = f"""Analyze the following text and identify the main topic for each page.
        Return ONLY a JSON object mapping page numbers to topics.
        Example format: {{"1": "Topic for page 1", "2": "Topic for page 2", ...}}

        Text content:
        {combined_text}

        Rules:
        1. Return ONLY the JSON object, no other text
        2. Each topic should be a short, descriptive phrase
        3. If a page has no clear topic, retry extract the topic for the page
        4. Make sure the JSON is properly formatted with double quotes
        5. Use page numbers as keys (1-based)
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response if it contains markdown code blocks
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse the response
        import json
        topics_dict = json.loads(response_text)
        
        # Convert string keys to integers and ensure all pages have topics
        result = {}
        for i in range(len(pages)):
            page_num = i + 1
            result[page_num] = topics_dict.get(str(page_num), "General Content")
        
        return result
        
    except Exception as e:
        logging.error(f"Error in topic extraction: {str(e)}")
        # Return "General Content" for all pages in case of error
        return {i+1: "General Content" for i in range(len(pages))}