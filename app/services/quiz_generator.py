import google.generativeai as genai
import os
import logging
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
            time.sleep(10)
        raise

async def generate_quiz_questions(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate multiple types of quiz questions based on the given text, topic, and chapter.
    Returns a dictionary containing different types of questions.
    """
    print(f"In generate_quiz_questions text: {text}")
    try:
        # Generate multiple choice questions
        mc_questions = await generate_multiple_choice(text)
        
        # Generate true/false questions
        # tf_questions = await generate_true_false(text)
        
        # # Generate fill in the blanks questions
        # fb_questions = await generate_fill_blanks(text)
        
        # # Generate short answer questions
        # sa_questions = await generate_short_answer(text)
        
        return {
            "multiple_choice": mc_questions,
            # "true_false": tf_questions,
            # "fill_blanks": fb_questions,
            # "short_answer": sa_questions
        }
    except Exception as e:
        logger.error(f"Error generating quiz questions: {str(e)}")
        raise Exception(f"Failed to generate quiz questions: {str(e)}")

async def generate_multiple_choice(text: str) -> List[Dict[str, Any]]:
    """Generate multiple choice questions"""
    try:
        prompt = f"""Generate 5 multiple-choice questions based on the following text.
        The questions should focus on the topic['name'] in topics list within the chapter['name'] in chapters list.
        
        Format the response as a JSON array of objects regarding per page with the following structure:
        [
            {{
                "question": "question text",
                "options": ["option1", "option2", "option3", "option4"],
                "correct_answer": "correct option",
                "explanation": "explanation of the correct answer",
                "page_number": page_number
            }}
        ]
        Make sure all backslashes in strings are properly escaped as double backslashes (\\), and the output is valid JSON.
        
        Text:
        {text}
        """
        
        response = await _generate_quiz_with_retry(prompt)
        print(f"In generate_multiple_choice response: {response.text}")
        return parse_questions(response.text)
    except Exception as e:
        logger.error(f"Error generating multiple choice questions: {str(e)}")
        return []

async def generate_true_false(text: str) -> List[Dict[str, Any]]:
    """Generate true/false questions"""
    try:

        prompt = f"""Generate 3 true/false questions based on the following text.
        The questions should focus on the topic['name'] in topics list within the chapter['name'] in chapters list.
        
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "question": "question text",
                "options": ["true", "false"],
                "correct_answer": "true or false",
                "explanation": "explanation of the correct answer",
                "page_number": page_number
            }}
        ]
        Make sure all backslashes in strings are properly escaped as double backslashes (\\), and the output is valid JSON.
        Text:
        {text}
        """
        
        response = await _generate_quiz_with_retry(prompt)
        return parse_questions(response.text)
    except Exception as e:
        logger.error(f"Error generating true/false questions: {str(e)}")
        return []

async def generate_fill_blanks(text: str) -> List[Dict[str, Any]]:
    """Generate fill in the blanks questions"""
    try:
        prompt = f"""Generate 3 fill in the blanks questions based on the following text.
        The questions should focus on the topic['name'] in topics list within the chapter['name'] in chapters list.
        
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "question": "sentence with _____ for the blank",
                "options": [],
                "correct_answer": "correct word or phrase",
                "explanation": "explanation of the correct answer",
                "page_number": page_number
            }}
        ]
        Make sure all backslashes in strings are properly escaped as double backslashes (\\), and the output is valid JSON.
        Text:
        {text}
        """
        
        response = await _generate_quiz_with_retry(prompt)
        return parse_questions(response.text)
    except Exception as e:
        logger.error(f"Error generating fill in the blanks questions: {str(e)}")
        return []

async def generate_short_answer(text: str) -> List[Dict[str, Any]]:
    """Generate short answer questions"""
    try:
        prompt = f"""Generate 2 short answer questions based on the following text.
        The questions should focus on the topic['name'] in topics list within the chapter['name'] in chapters list.
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "question": "question text",
                "options": [],
                "correct_answer": "model answer",
                "explanation": "explanation of the answer",
                "page_number": page_number
            }}
        ]
        
        Make sure all backslashes in strings are properly escaped as double backslashes (\\), and the output is valid JSON.
        Text:
        {text}
        """
        
        response = await _generate_quiz_with_retry(prompt)
        return parse_questions(response.text)
    except Exception as e:
        logger.error(f"Error generating short answer questions: {str(e)}")
        return []

def parse_questions(text: str) -> List[Dict[str, Any]]:
    """
    Parse the AI response into a structured list of questions.
    """
    try:
        import json
        import re
        
        # Find JSON array in the text
        json_match = re.search(r'\[[\s\S]*\]', text)
        if not json_match:
            return []
        
        json_text = json_match.group()
        # Sanitize bad backslashes (escape invalid ones)
        sanitized_text = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', json_text)
        logger.info(f"In parse_questions sanitized_text: {sanitized_text}")
        questions = json.loads(sanitized_text)
        logger.info(f"In parse_questions questions: {questions}")

        # Validate question structure
        valid_questions = []
        for question in questions:
            if (isinstance(question, dict) and 
                'question' in question and 
                'options' in question and 
                'correct_answer' in question and 
                'explanation' in question):
                
                # For multiple choice questions, ensure options is a list with exactly 4 items
                if isinstance(question['options'], list):
                    if len(question['options']) == 4:
                        if question['correct_answer'] in question['options']:
                            valid_questions.append(question)
                    elif len(question['options']) == 2 and question['options'] == ['true', 'false']:
                        valid_questions.append(question)
                    elif len(question['options']) == 0:
                        valid_questions.append(question)
        print('-------- valide questions ----------\n', questions)
        return valid_questions
    except Exception as e:
        logger.error(f"Error parsing questions: {str(e)}")
        return [] 