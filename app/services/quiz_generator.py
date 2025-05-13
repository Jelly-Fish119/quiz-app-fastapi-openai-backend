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


async def generate_quiz_questions(text: str, chapters: List[Dict[str, Any]], topics: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate multiple types of quiz questions based on the given text, topic, and chapter.
    Returns a dictionary containing different types of questions.
    """
    try:
        # Generate multiple choice questions
        mc_questions = await generate_multiple_choice(text, topics, chapters)
        
        # Generate true/false questions
        tf_questions = await generate_true_false(text, topics, chapters)
        
        # Generate fill in the blanks questions
        fb_questions = await generate_fill_blanks(text, topics, chapters)
        
        # Generate short answer questions
        sa_questions = await generate_short_answer(text, topics, chapters)
        
        return {
            "multiple_choice": mc_questions,
            "true_false": tf_questions,
            "fill_blanks": fb_questions,
            "short_answer": sa_questions
        }
    except Exception as e:
        logger.error(f"Error generating quiz questions: {str(e)}")
        raise Exception(f"Failed to generate quiz questions: {str(e)}")

async def generate_multiple_choice(text: str, topics: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate multiple choice questions"""
    try:
        topics_text = ','.join([topic.name for topic in topics])
        chapters_text = ','.join([chapter.name for chapter in chapters])
        logger.info(f"""chapters_text: {chapters_text}
                     topics_text: {topics_text}""")
        prompt = f"""Generate 5 multiple-choice questions based on the following text.
        The questions should focus on the topics: {topics_text} within the chapters: {chapters_text}.
        
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "question": "question text",
                "options": ["option1", "option2", "option3", "option4"],
                "correct_answer": "correct option",
                "explanation": "explanation of the correct answer"
            }}
        ]
        
        Text:
        {text}
        """
        
        response = await model.generate_content(prompt)
        return parse_questions(response.text)
    except Exception as e:
        logger.error(f"Error generating multiple choice questions: {str(e)}")
        return []

async def generate_true_false(text: str, topics: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate true/false questions"""
    logger.info(f"""chapters: {chapters}, 
                 topics: {topics}""")
    try:
        topics_text = ','.join([topic.name for topic in topics])
        chapters_text = ','.join([chapter.name for chapter in chapters])

        prompt = f"""Generate 3 true/false questions based on the following text.
        The questions should focus on the topics: {topics_text} within the chapters: {chapters_text}.
        
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "question": "question text",
                "options": ["true", "false"],
                "correct_answer": "true or false",
                "explanation": "explanation of the correct answer"
            }}
        ]
        
        Text:
        {text}
        """
        
        response = await model.generate_content(prompt)
        return parse_questions(response.text)
    except Exception as e:
        logger.error(f"Error generating true/false questions: {str(e)}")
        return []

async def generate_fill_blanks(text: str, topics: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate fill in the blanks questions"""
    try:
        topics_text = ','.join([topic.name for topic in topics])
        chapters_text = ','.join([chapter.name for chapter in chapters])
        prompt = f"""Generate 3 fill in the blanks questions based on the following text.
        The questions should focus on the topics: {topics_text} within the chapters: {chapters_text}.
        
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "question": "sentence with _____ for the blank",
                "options": [],
                "correct_answer": "correct word or phrase",
                "explanation": "explanation of the correct answer"
            }}
        ]
        
        Text:
        {text}
        """
        
        response = await model.generate_content(prompt)
        return parse_questions(response.text)
    except Exception as e:
        logger.error(f"Error generating fill in the blanks questions: {str(e)}")
        return []

async def generate_short_answer(text: str, topics: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate short answer questions"""
    try:
        topics_text = ','.join([topic.name for topic in topics])
        chapters_text = ','.join([chapter.name for chapter in chapters])
        prompt = f"""Generate 2 short answer questions based on the following text.
        The questions should focus on the topics: {topics_text} within the chapters: {chapters_text}.
        
        Format the response as a JSON array of objects with the following structure:
        [
            {{
                "question": "question text",
                "options": [],
                "correct_answer": "model answer",
                "explanation": "explanation of the answer"
            }}
        ]
        
        Text:
        {text}
        """
        
        response = await model.generate_content(prompt)
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
        
        questions = json.loads(json_match.group())
        
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
        
        return valid_questions
    except Exception as e:
        logger.error(f"Error parsing questions: {str(e)}")
        return [] 