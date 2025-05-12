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

async def generate_quiz_questions(texts: List[str], topics: List[Dict[str, Any]], chapters: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    try:
        # Combine all text
        combined_text = "\n\n".join(texts)
        
        # Create context from topics and chapters
        context = f"Topics: {', '.join(t['name'] for t in topics)}\n"
        context += f"Chapters: {', '.join(c['name'] for c in chapters)}\n\n"
        context += "Text:\n" + combined_text

        # Generate multiple choice questions
        mc_prompt = f"""Based on the following text, generate 5 multiple choice questions. Each question should have 4 options and one correct answer.
        Include an explanation for each correct answer.
        Format each question as a JSON object with the following structure:
        {{
            "question": "question text",
            "options": ["option1", "option2", "option3", "option4"],
            "correct_answer": "correct option",
            "explanation": "explanation of the correct answer",
            "type": "multiple_choice"
        }}
        
        Text:
        {context}"""

        mc_response = await model.generate_content(mc_prompt)
        multiple_choice = parse_questions(mc_response.text)

        # Generate true/false questions
        tf_prompt = f"""Based on the following text, generate 3 true/false questions.
        Include an explanation for each answer.
        Format each question as a JSON object with the following structure:
        {{
            "question": "question text",
            "options": ["true", "false"],
            "correct_answer": "true or false",
            "explanation": "explanation of the correct answer",
            "type": "true_false"
        }}
        
        Text:
        {context}"""

        tf_response = await model.generate_content(tf_prompt)
        true_false = parse_questions(tf_response.text)

        # Generate fill in the blanks questions
        fb_prompt = f"""Based on the following text, generate 3 fill in the blanks questions.
        Each question should have a sentence with a blank space and the correct answer.
        Format each question as a JSON object with the following structure:
        {{
            "question": "sentence with _____ for the blank",
            "options": [],
            "correct_answer": "correct word or phrase",
            "explanation": "explanation of the correct answer",
            "type": "fill_blanks"
        }}
        
        Text:
        {context}"""

        fb_response = await model.generate_content(fb_prompt)
        fill_blanks = parse_questions(fb_response.text)

        # Generate matching questions
        match_prompt = f"""Based on the following text, generate 2 matching questions.
        Each question should have a list of terms and their definitions or related items.
        Format each question as a JSON object with the following structure:
        {{
            "question": "Match the following terms with their definitions:",
            "options": ["term1", "term2", "term3", "term4"],
            "correct_answer": "definition1,definition2,definition3,definition4",
            "explanation": "explanation of the correct matches",
            "type": "matching"
        }}
        
        Text:
        {context}"""

        match_response = await model.generate_content(match_prompt)
        matching = parse_questions(match_response.text)

        # Generate short answer questions
        sa_prompt = f"""Based on the following text, generate 2 short answer questions.
        Include a model answer and explanation for each question.
        Format each question as a JSON object with the following structure:
        {{
            "question": "question text",
            "options": [],
            "correct_answer": "model answer",
            "explanation": "explanation of the answer",
            "type": "short_answer"
        }}
        
        Text:
        {context}"""

        sa_response = await model.generate_content(sa_prompt)
        short_answer = parse_questions(sa_response.text)

        return {
            "multiple_choice": multiple_choice,
            "true_false": true_false,
            "fill_blanks": fill_blanks,
            "matching": matching,
            "short_answer": short_answer
        }

    except Exception as e:
        logger.error(f"Error generating quiz questions: {str(e)}")
        raise Exception(f"Failed to generate quiz questions: {str(e)}")

def parse_questions(text: str) -> List[Dict[str, Any]]:
    try:
        # Extract JSON objects from the text
        import json
        import re
        
        # Find all JSON objects in the text
        json_objects = re.findall(r'\{[^{}]*\}', text)
        
        questions = []
        for obj in json_objects:
            try:
                question = json.loads(obj)
                if all(key in question for key in ['question', 'options', 'correct_answer', 'explanation', 'type']):
                    questions.append(question)
            except json.JSONDecodeError:
                continue
        
        return questions
    except Exception as e:
        logger.error(f"Error parsing questions: {str(e)}")
        return [] 