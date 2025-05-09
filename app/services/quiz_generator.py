from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
from typing import List, Dict
import logging
import json

genai.configure(api_key=GOOGLE_API_KEY)

def generate_quiz_questions(text: str, topic: str) -> List[Dict[str, str]]:
    """
    Generate quiz questions based on the given text and topic.
    Returns a list of dictionaries containing questions and their answers.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a quiz generator. Create exactly 3 multiple choice questions based on the following text.
        The text is about: {topic}

        Text content:
        {text}

        IMPORTANT: You must respond with ONLY a valid JSON array containing exactly 3 questions.
        Each question must have exactly this structure:
        {{
            "question": "The question text",
            "options": {{
                "A": "First option",
                "B": "Second option",
                "C": "Third option",
                "D": "Fourth option"
            }},
            "correct_answer": "A"
        }}

        Rules:
        1. Return ONLY the JSON array, no other text
        2. Each question must have exactly 4 options (A, B, C, D)
        3. The correct_answer must be one of: A, B, C, or D
        4. Make sure the JSON is properly formatted with double quotes
        5. Do not include any explanations or additional text
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Try to clean the response if it contains markdown code blocks
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            questions = json.loads(response_text)
            # Validate the structure
            if not isinstance(questions, list) or len(questions) != 3:
                raise ValueError("Invalid number of questions")
            
            for q in questions:
                if not all(k in q for k in ['question', 'options', 'correct_answer']):
                    raise ValueError("Invalid question structure")
                if not all(k in q['options'] for k in ['A', 'B', 'C', 'D']):
                    raise ValueError("Invalid options structure")
                if q['correct_answer'] not in ['A', 'B', 'C', 'D']:
                    raise ValueError("Invalid correct answer")
            
            return questions
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse quiz questions: {str(e)}")
            logging.error(f"Raw response: {response_text}")
            return [{
                "question": "Failed to generate questions",
                "options": {
                    "A": "Error occurred",
                    "B": "Please try again",
                    "C": "Contact support",
                    "D": "None of the above"
                },
                "correct_answer": "A"
            }]
    except Exception as e:
        logging.error(f"Error in quiz generation: {str(e)}")
        return [{
            "question": "Failed to generate questions",
            "options": {
                "A": "Error occurred",
                "B": "Please try again",
                "C": "Contact support",
                "D": "None of the above"
            },
            "correct_answer": "A"
        }] 