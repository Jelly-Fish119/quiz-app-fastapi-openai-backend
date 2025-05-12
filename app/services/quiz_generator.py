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

async def generate_quiz_questions(text: str, topic: str, chapter: str) -> List[Dict[str, Any]]:
    try:
        prompt = f"""Generate 5 multiple-choice questions based on the following text.
        The questions should focus on the topic "{topic}" within the chapter "{chapter}".
        
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
        questions = parse_questions(response.text)
        
        if not questions:
            raise Exception("Failed to generate valid questions")
        
        return questions
    except Exception as e:
        logger.error(f"Error generating quiz questions: {str(e)}")
        raise Exception(f"Failed to generate quiz questions: {str(e)}")

def parse_questions(text: str) -> List[Dict[str, Any]]:
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
                
                # Ensure options is a list with exactly 4 items
                if (isinstance(question['options'], list) and 
                    len(question['options']) == 4 and 
                    question['correct_answer'] in question['options']):
                    
                    valid_questions.append({
                        'question': question['question'],
                        'options': question['options'],
                        'correct_answer': question['correct_answer'],
                        'explanation': question['explanation']
                    })
        
        return valid_questions
    except Exception as e:
        logger.error(f"Error parsing questions: {str(e)}")
        return [] 