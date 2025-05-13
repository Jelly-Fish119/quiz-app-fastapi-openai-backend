import google.generativeai as genai
import os
import logging
import json
import re
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
async def _generate_quiz_with_retry(prompt: str) -> str:
    try:
        response = await model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            logger.warning(f"Rate limit hit, will retry: {str(e)}")
            raise RateLimitError(str(e))
        raise

def extract_json_section(response: str, key: str) -> List[Dict[str, Any]]:
    try:
        match = re.search(rf'"{key}"\s*:\s*(\[\s*\{{[\s\S]*?\}}\s*\])', response)
        if match:
            json_text = match.group(1)
            sanitized = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', json_text)
            return json.loads(sanitized)
    except Exception as e:
        logger.error(f"Error extracting section '{key}': {e}")
    return []

async def generate_quiz_questions(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate quiz questions for each page (MCQ, True/False, Fill in the Blanks, Short Answer)
    using a single Gemini API call.
    """
    import json

    try:
        prompt = f"""
        You are a quiz generator. Based on the following JSON data, generate quiz questions **per page**.
        Each element in the list represents a single page with associated text, topics, and chapter context.

        For each page:
        - Use the `text` to generate:
        - 5 Multiple Choice Questions (MCQs)
        - 3 True/False Questions
        - 3 Fill in the Blanks Questions
        - 2 Short Answer Questions
        - Focus on the topic['name'] from the `topics` list within the chapter['name'] in the `chapters` list.
        - Structure your output as a **JSON array**, where each item corresponds to a page with the following format:

        [
        {{
            "page_number": <page_number>,
            "multiple_choice": [
            {{
                "question": "...",
                "options": ["...", "...", "...", "..."],
                "correct_answer": "...",
                "explanation": "..."
            }},
            ...
            ],
            "true_false": [
            {{
                "question": "...",
                "options": ["true", "false"],
                "correct_answer": "...",
                "explanation": "..."
            }},
            ...
            ],
            "fill_in_the_blanks": [
            {{
                "question": "Fill in the blank: ... _____ ...",
                "options": [],
                "correct_answer": "...",
                "explanation": "..."
            }},
            ...
            ],
            "short_answer": [
            {{
                "question": "...",
                "options": [],
                "correct_answer": "...",
                "explanation": "..."
            }},
            ...
            ]
        }},
        ...
        ]

        Make sure the JSON is valid and properly escapes any backslashes (\\).
        Here is the input JSON data:
        {json.dumps(pages, indent=2)}
        """
        response = await _generate_quiz_with_retry(prompt)
        return parse_full_page_quiz(response)
    except Exception as e:
        logger.error(f"Error generating full quiz questions: {str(e)}")
        raise


def parse_full_page_quiz(text: str) -> List[Dict[str, Any]]:
    """
    Parses the Gemini response into a list of per-page structured quiz data.
    """
    import json
    import re

    try:
        # Extract JSON array from response
        json_match = re.search(r'\[[\s\S]*\]', text)
        if not json_match:
            raise ValueError("No JSON array found in response.")

        json_text = json_match.group()

        # Sanitize unescaped backslashes
        sanitized_text = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', json_text)
        pages = json.loads(sanitized_text)

        valid_pages = []
        for page in pages:
            if 'page_number' not in page:
                continue
            for section in ['multiple_choice', 'true_false', 'fill_in_the_blanks', 'short_answer']:
                if section not in page:
                    page[section] = []
            valid_pages.append(page)

        return valid_pages
    except Exception as e:
        logger.error(f"Error parsing full page quiz response: {str(e)}")
        return []
