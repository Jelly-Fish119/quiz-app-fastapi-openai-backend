from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
from typing import List, Dict
import logging

genai.configure(api_key=GOOGLE_API_KEY)

def generate_quiz_questions(text: str, topic: str) -> List[Dict[str, str]]:
    """
    Generate quiz questions based on the given text and topic.
    Returns a list of dictionaries containing questions and their answers.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Based on the following text about {topic}, generate 3 multiple choice questions.
        For each question, provide 4 options (A, B, C, D) and indicate the correct answer.
        Format each question as a JSON object with 'question', 'options', and 'correct_answer' fields.
        
        Text:
        {text}
        
        Return the questions in this format:
        [
            {{
                "question": "Question text here?",
                "options": {{
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D"
                }},
                "correct_answer": "A"
            }}
        ]
        """
        
        response = model.generate_content(prompt)
        try:
            # Parse the response text as JSON
            import json
            questions = json.loads(response.text)
            return questions
        except json.JSONDecodeError:
            logging.error("Failed to parse quiz questions as JSON")
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