import google.generativeai as genai
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.0-pro')


async def analyze_chapters(text: str) -> List[Dict[str, Any]]:
    """
    Analyze the given text to identify chapters or sections.
    Returns a list of chapters with their confidence scores.
    """
    try:
        prompt = f"""Analyze the following text and identify the main chapters or sections.
        For each chapter, provide a confidence score between 0 and 1.
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "name": "chapter name",
                "confidence": confidence_score
            }}
        ]
        
        Text:
        {text}
        """
        
        response = await model.generate_content(prompt)
        chapters = parse_chapters(response.text)
        
        if not chapters:
            logger.warning("No chapters identified in text")
            return [{"name": "General Content", "confidence": 1.0}]
        
        return chapters
    except Exception as e:
        logger.error(f"Error analyzing chapters: {str(e)}")
        return [{"name": "General Content", "confidence": 1.0}]

def parse_chapters(text: str) -> List[Dict[str, Any]]:
    """
    Parse the AI response into a structured list of chapters.
    """
    try:
        import json
        import re
        
        # Find JSON array in the text
        json_match = re.search(r'\[[\s\S]*\]', text)
        if not json_match:
            return []
        
        chapters = json.loads(json_match.group())
        
        # Validate chapter structure
        valid_chapters = []
        for chapter in chapters:
            if isinstance(chapter, dict) and 'name' in chapter and 'confidence' in chapter:
                # Ensure confidence is a float between 0 and 1
                confidence = float(chapter['confidence'])
                if 0 <= confidence <= 1:
                    valid_chapters.append({
                        'name': chapter['name'],
                        'confidence': confidence
                    })
        
        return valid_chapters
    except Exception as e:
        logger.error(f"Error parsing chapters: {str(e)}")
        return [] 