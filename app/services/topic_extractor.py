import google.generativeai as genai
import os
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

async def extract_topics(text: str) -> List[Dict[str, Any]]:
    """
    Extract topics from the given text using Gemini AI.
    Returns a list of topics with their confidence scores.
    """
    try:
        prompt = f"""Analyze the following text and identify the main topics or themes.
        For each topic, provide a confidence score between 0 and 1.
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "name": "topic name",
                "confidence": confidence_score
            }}
        ]
        
        Text:
        {text}
        """
        
        response = await model.generate_content(prompt)
        topics = parse_topics(response.text)
        
        if not topics:
            logger.warning("No topics extracted from text")
            return [{"name": "General Knowledge", "confidence": 1.0}]
        
        return topics
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        return [{"name": "General Knowledge", "confidence": 1.0}]

def parse_topics(text: str) -> List[Dict[str, Any]]:
    """
    Parse the AI response into a structured list of topics.
    """
    try:
        import json
        import re
        
        # Find JSON array in the text
        json_match = re.search(r'\[[\s\S]*\]', text)
        if not json_match:
            return []
        
        topics = json.loads(json_match.group())
        
        # Validate topic structure
        valid_topics = []
        for topic in topics:
            if isinstance(topic, dict) and 'name' in topic and 'confidence' in topic:
                # Ensure confidence is a float between 0 and 1
                confidence = float(topic['confidence'])
                if 0 <= confidence <= 1:
                    valid_topics.append({
                        'name': topic['name'],
                        'confidence': confidence
                    })
        
        return valid_topics
    except Exception as e:
        logger.error(f"Error parsing topics: {str(e)}")
        return [] 