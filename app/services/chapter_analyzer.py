import google.generativeai as genai
import os
import logging
import asyncio
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro')

class RateLimitError(Exception):
    """Custom exception for rate limiting errors"""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True
)
async def _generate_content_with_retry(prompt: str) -> str:
    """
    Generate content with retry logic for rate limiting.
    """
    try:
        response = await model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            logger.warning(f"Rate limit hit, will retry: {str(e)}")
            raise RateLimitError(str(e))
        raise

async def analyze_chapters(text: str) -> List[Dict[str, Any]]:
    """
    Analyze the given text to identify chapters or sections.
    Returns a list of chapters with their confidence scores.
    """
    try:
        # First, check if the text is meaningful
        if not text or len(text.strip()) < 10:
            logger.error("Text is too short or empty")
            raise ValueError("Text is too short or empty")

        prompt = f"""You are an expert at analyzing educational content and identifying chapters or sections.
        Analyze the following text and identify the main chapters or sections.
        Focus on extracting specific, detailed chapter names rather than general ones.
        
        Guidelines:
        1. Identify 2-3 most relevant chapters or sections
        2. Each chapter should be specific and detailed (e.g., "Introduction to Neural Networks" instead of "Introduction")
        3. Chapters should be relevant to the content
        4. Avoid generic chapters like "General Content" or "General Chapter" or "General Knowledge"
        5. Consider the context and domain of the text
        6. If you can't identify specific chapters, return an empty array instead of generic chapters
        
        For each chapter, provide a confidence score between 0 and 1.
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "name": "specific chapter name",
                "confidence": confidence_score
            }}
        ]
        
        Text to analyze:
        {text}
        """
        
        response_text = await _generate_content_with_retry(prompt)
        chapters = parse_chapters(response_text)
        
        if not chapters:
            logger.warning("No chapters identified in text, trying with a more specific prompt")
            # Try with a more specific prompt if the first attempt fails
            specific_prompt = f"""Extract specific chapters or sections from this educational content.
            Focus on identifying clear chapter divisions and section headers.
            Avoid generic chapter names.
            If you can't identify specific chapters, return an empty array.
            
            Text:
            {text}
            
            Return as JSON array:
            [
                {{
                    "name": "specific chapter",
                    "confidence": score
                }}
            ]
            """
            
            response_text = await _generate_content_with_retry(specific_prompt)
            chapters = parse_chapters(response_text)
        
        if not chapters:
            logger.error("Failed to identify specific chapters")
            raise ValueError("Could not identify meaningful chapters from the text")
        
        # Filter out low confidence chapters and sort by confidence
        chapters = [c for c in chapters if c['confidence'] >= 0.6]
        chapters.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not chapters:
            raise ValueError("No chapters met the confidence threshold")
        
        return chapters
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded after retries: {str(e)}")
        raise Exception("API rate limit exceeded. Please try again in a few minutes.")
    except Exception as e:
        logger.error(f"Error analyzing chapters: {str(e)}")
        raise Exception(f"Failed to analyze chapters: {str(e)}")

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
                    # Skip generic chapters
                    if chapter['name'].lower() not in ['general content', 'general chapter', 'general section']:
                        valid_chapters.append({
                            'name': chapter['name'],
                            'confidence': confidence
                        })
        
        return valid_chapters
    except Exception as e:
        logger.error(f"Error parsing chapters: {str(e)}")
        return [] 