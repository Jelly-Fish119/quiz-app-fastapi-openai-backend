from fastapi import APIRouter, HTTPException
from app.services.topic_extractor import extract_topics
from app.services.chapter_analyzer import analyze_chapters
from app.services.quiz_generator import generate_quiz_questions
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
import google.generativeai as genai
import os
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

router = APIRouter()

class Page(BaseModel):
    page_number: int
    text: str

class PageAnalysisRequest(BaseModel):
    pages: List[Page]

class QuizGenerationRequest(BaseModel):
    page_number: int
    text: str
    topic: str
    chapter: str

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text, handling potential formatting issues.
    """
    try:
        # First try direct JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Try to find JSON object in the text
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON object found in response")
        except Exception as e:
            logger.error(f"Failed to extract JSON from response: {text}")
            raise ValueError(f"Invalid response format: {str(e)}")

async def analyze_all_pages(pages: List[Page]) -> Dict[str, Any]:
    """
    Analyze all pages' content in a single call to extract topics and chapters.
    """
    try:
        # Combine all pages' text with page numbers
        combined_text = "\n\n".join([
            f"=== Page {page.page_number} ===\n{page.text}"
            for page in pages
        ])

        prompt = f"""You are an expert at analyzing educational content.
        Analyze the following text from multiple pages and identify topics and chapters for each page.
        
        Guidelines for Topics:
        1. Identify 3-5 most important topics per page
        2. Each topic should be specific and detailed
        3. Topics should be relevant to the content
        4. Avoid generic topics
        5. Provide a confidence score between 0 and 1 for each topic
        
        Guidelines for Chapters:
        1. Identify 2-3 most relevant chapters or sections per page
        2. Each chapter should be specific and detailed
        3. Chapters should be relevant to the content
        4. Avoid generic chapter names
        5. Provide a confidence score between 0 and 1 for each chapter
        
        IMPORTANT: Respond ONLY with a valid JSON object in the following format, with no additional text:
        {{
            "pages": [
                {{
                    "page_number": page_number,
                    "topics": [
                        {{
                            "name": "specific topic name",
                            "confidence": confidence_score
                        }}
                    ],
                    "chapters": [
                        {{
                            "name": "specific chapter name",
                            "confidence": confidence_score
                        }}
                    ]
                }}
            ]
        }}
        
        Text to analyze (each section is marked with page numbers):
        {combined_text}
        """
        
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Empty response from model")
            
        logger.info(f"Raw model response: {response.text}")
        result = extract_json_from_text(response.text)
        
        # Validate the response structure
        if not isinstance(result, dict) or 'pages' not in result:
            raise ValueError("Invalid response structure: missing 'pages' key")
            
        # Filter out low confidence items and sort by confidence for each page
        for page_result in result['pages']:
            if not isinstance(page_result, dict):
                continue
                
            # Ensure required fields exist
            page_result.setdefault('topics', [])
            page_result.setdefault('chapters', [])
            
            # Filter and sort topics
            page_result['topics'] = [
                t for t in page_result['topics'] 
                if isinstance(t, dict) and 
                'name' in t and 
                'confidence' in t and 
                isinstance(t['confidence'], (int, float)) and
                t['confidence'] >= 0.6
            ]
            page_result['topics'].sort(key=lambda x: x['confidence'], reverse=True)
            
            # Filter and sort chapters
            page_result['chapters'] = [
                c for c in page_result['chapters'] 
                if isinstance(c, dict) and 
                'name' in c and 
                'confidence' in c and 
                isinstance(c['confidence'], (int, float)) and
                c['confidence'] >= 0.6
            ]
            page_result['chapters'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing pages content: {str(e)}")
        raise Exception(f"Failed to analyze pages content: {str(e)}")

@router.post("/pdf/analyze-pages")
async def analyze_pages(request: PageAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze all pages in a single call to extract topics and chapters.
    """
    try:
        # Analyze all pages content in one call
        analysis = await analyze_all_pages(request.pages)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing pages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while analyzing pages: {str(e)}"
        )

@router.post("/pdf/generate-quiz")
async def generate_quiz(request: QuizGenerationRequest) -> Dict[str, Any]:
    """
    Generate quiz questions for a specific page.
    """
    try:
        # Generate quiz questions
        questions = await generate_quiz_questions(
            request.text,
            request.topic,
            request.chapter
        )
        
        return {
            "page_number": request.page_number,
            "topic": request.topic,
            "chapter": request.chapter,
            "questions": questions
        }
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating quiz: {str(e)}"
        ) 
