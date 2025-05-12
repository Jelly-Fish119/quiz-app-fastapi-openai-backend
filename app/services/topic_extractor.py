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

async def extract_topics(text: str) -> List[Dict[str, Any]]:
    """
    Extract topics from the given text using Gemini AI.
    Returns a list of topics with their confidence scores.
    """
    try:
        # First, check if the text is meaningful
        if not text or len(text.strip()) < 10:
            logger.error("Text is too short or empty")
            raise ValueError("Text is too short or empty")

        prompt = f"""You are an expert at analyzing educational content and identifying specific topics.
        Analyze the following text and identify the main topics or themes.
        Focus on extracting specific, detailed topics rather than general ones.
        
        Guidelines:
        1. Identify 3-5 most important topics
        2. Each topic should be specific and detailed (e.g., "Neural Network Architecture" instead of "Machine Learning")
        3. Topics should be relevant to the content
        4. Avoid generic topics like "General Knowledge" or "General Content"
        5. Consider the context and domain of the text
        6. If you can't identify specific topics, return an empty array instead of generic topics
        
        For each topic, provide a confidence score between 0 and 1.
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "name": "specific topic name",
                "confidence": confidence_score
            }}
        ]
        
        Text to analyze:
        {text}
        """
        
        response_text = await _generate_content_with_retry(prompt)
        topics = parse_topics(response_text)
        
        if not topics:
            logger.warning("No topics extracted from text, trying with a more specific prompt")
            # Try with a more specific prompt if the first attempt fails
            specific_prompt = f"""Extract specific topics from this educational content.
            Focus on technical terms, concepts, and specific subject areas.
            Avoid general topics.
            If you can't identify specific topics, return an empty array.
            
            Text:
            {text}
            
            Return as JSON array:
            [
                {{
                    "name": "specific topic",
                    "confidence": score
                }}
            ]
            """
            
            response_text = await _generate_content_with_retry(specific_prompt)
            topics = parse_topics(response_text)
        
        if not topics:
            logger.error("Failed to extract specific topics")
            raise ValueError("Could not extract meaningful topics from the text")
        
        # Filter out low confidence topics and sort by confidence
        topics = [t for t in topics if t['confidence'] >= 0.6]
        topics.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not topics:
            raise ValueError("No topics met the confidence threshold")
        
        return topics
    except RateLimitError as e:
        logger.error(f"Rate limit exceeded after retries: {str(e)}")
        raise Exception("API rate limit exceeded. Please try again in a few minutes.")
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        raise Exception(f"Failed to extract topics: {str(e)}")

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
                    # Skip generic topics
                    if topic['name'].lower() not in ['general knowledge', 'general content', 'general topic']:
                        valid_topics.append({
                            'name': topic['name'],
                            'confidence': confidence
                        })
        
        return valid_topics
    except Exception as e:
        logger.error(f"Error parsing topics: {str(e)}")
        return [] 