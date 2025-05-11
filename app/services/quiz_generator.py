from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
from typing import List, Dict, Union
import logging
import json

genai.configure(api_key=GOOGLE_API_KEY)

QuestionType = Union[
    Dict[str, str],  # Multiple Choice
    Dict[str, List[str]],  # Fill in the Blanks
    Dict[str, bool],  # True/False
    Dict[str, Dict[str, str]]  # Matching
]

def generate_quiz_questions(text: str, topic: str, chapter: str) -> Dict[str, List[QuestionType]]:
    """
    Generate different types of quiz questions based on the given text and topic.
    Returns a dictionary containing lists of different question types.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a quiz generator. Create questions based on the following text.
        The text is about: {topic} and {chapter}

        Text content:
        {text}

        Generate the following types of questions:
        1. Multiple Choice (3 questions)
        2. Fill in the Blanks (2 questions)
        3. True/False (2 questions)
        4. Matching (1 question with 4 pairs)

        IMPORTANT: You must respond with ONLY a valid JSON object containing all question types.
        Use this exact structure:
        {{
            "multiple_choice": [
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
            ],
            "fill_blanks": [
                {{
                    "question": "Complete the sentence: [BLANK] is important because [BLANK]",
                    "answers": ["First blank answer", "Second blank answer"]
                }}
            ],
            "true_false": [
                {{
                    "question": "Statement to evaluate",
                    "correct_answer": true
                }}
            ],
            "matching": [
                {{
                    "question": "Match the following:",
                    "pairs": {{
                        "Term 1": "Definition 1",
                        "Term 2": "Definition 2",
                        "Term 3": "Definition 3",
                        "Term 4": "Definition 4"
                    }}
                }}
            ]
        }}

        Rules:
        1. Return ONLY the JSON object, no other text
        2. Multiple choice questions must have exactly 4 options (A, B, C, D)
        3. Fill in the blanks questions should have [BLANK] markers
        4. True/False questions must have boolean answers
        5. Matching questions must have exactly 4 pairs
        6. Make sure the JSON is properly formatted with double quotes
        7. Do not include any explanations or additional text
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
            required_sections = ['multiple_choice', 'fill_blanks', 'true_false', 'matching']
            if not all(section in questions for section in required_sections):
                raise ValueError("Missing required question sections")
            
            # Validate multiple choice questions
            if not isinstance(questions['multiple_choice'], list) or len(questions['multiple_choice']) != 3:
                raise ValueError("Invalid number of multiple choice questions")
            for q in questions['multiple_choice']:
                if not all(k in q for k in ['question', 'options', 'correct_answer']):
                    raise ValueError("Invalid multiple choice question structure")
                if not all(k in q['options'] for k in ['A', 'B', 'C', 'D']):
                    raise ValueError("Invalid multiple choice options structure")
                if q['correct_answer'] not in ['A', 'B', 'C', 'D']:
                    raise ValueError("Invalid multiple choice correct answer")
            
            # Validate fill in the blanks questions
            if not isinstance(questions['fill_blanks'], list) or len(questions['fill_blanks']) != 2:
                raise ValueError("Invalid number of fill in the blanks questions")
            for q in questions['fill_blanks']:
                if not all(k in q for k in ['question', 'answers']):
                    raise ValueError("Invalid fill in the blanks question structure")
                if not isinstance(q['answers'], list):
                    raise ValueError("Fill in the blanks answers must be a list")
            
            # Validate true/false questions
            if not isinstance(questions['true_false'], list) or len(questions['true_false']) != 2:
                raise ValueError("Invalid number of true/false questions")
            for q in questions['true_false']:
                if not all(k in q for k in ['question', 'correct_answer']):
                    raise ValueError("Invalid true/false question structure")
                if not isinstance(q['correct_answer'], bool):
                    raise ValueError("True/false answer must be boolean")
            
            # Validate matching questions
            if not isinstance(questions['matching'], list) or len(questions['matching']) != 1:
                raise ValueError("Invalid number of matching questions")
            for q in questions['matching']:
                if not all(k in q for k in ['question', 'pairs']):
                    raise ValueError("Invalid matching question structure")
                if not isinstance(q['pairs'], dict) or len(q['pairs']) != 4:
                    raise ValueError("Matching question must have exactly 4 pairs")
            
            return questions
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse quiz questions: {str(e)}")
            logging.error(f"Raw response: {response_text}")
            return {
                "multiple_choice": [{
                    "question": "Failed to generate questions",
                    "options": {
                        "A": "Error occurred",
                        "B": "Please try again",
                        "C": "Contact support",
                        "D": "None of the above"
                    },
                    "correct_answer": "A"
                }],
                "fill_blanks": [{
                    "question": "Failed to generate questions",
                    "answers": ["Error occurred"]
                }],
                "true_false": [{
                    "question": "Failed to generate questions",
                    "correct_answer": True
                }],
                "matching": [{
                    "question": "Failed to generate questions",
                    "pairs": {
                        "Error": "Error occurred",
                        "Please": "Try again",
                        "Contact": "Support",
                        "None": "Of the above"
                    }
                }]
            }
    except Exception as e:
        logging.error(f"Error in quiz generation: {str(e)}")
        return {
            "multiple_choice": [{
                "question": "Failed to generate questions",
                "options": {
                    "A": "Error occurred",
                    "B": "Please try again",
                    "C": "Contact support",
                    "D": "None of the above"
                },
                "correct_answer": "A"
            }],
            "fill_blanks": [{
                "question": "Failed to generate questions",
                "answers": ["Error occurred"]
            }],
            "true_false": [{
                "question": "Failed to generate questions",
                "correct_answer": True
            }],
            "matching": [{
                "question": "Failed to generate questions",
                "pairs": {
                    "Error": "Error occurred",
                    "Please": "Try again",
                    "Contact": "Support",
                    "None": "Of the above"
                }
            }]
        } 