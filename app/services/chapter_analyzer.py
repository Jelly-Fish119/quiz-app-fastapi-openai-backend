from google import generativeai as genai
from app.core.config import GOOGLE_API_KEY
from typing import List, Dict, Any
import logging
import json

genai.configure(api_key=GOOGLE_API_KEY)

def analyze_chapters(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze the text to identify chapters and their topics with page numbers and positions.
    Returns a dictionary containing chapters and their associated topics.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""You are a document analyzer. Analyze the following text to identify chapters and their topics.
        For each chapter and topic, provide:
        1. Title/name
        2. Page number
        3. Position on the page (top and left coordinates as percentages)
        4. For topics, include any sub-topics

        Text content:
        {text}

        IMPORTANT: You must respond with ONLY a valid JSON object containing all chapters and topics.
        Use this exact structure:
        {{
            "chapters": [
                {{
                    "title": "Chapter Title",
                    "pageNumber": 1,
                    "position": {{
                        "top": 10,
                        "left": 20
                    }},
                    "topics": [
                        {{
                            "name": "Topic Name",
                            "pageNumber": 1,
                            "position": {{
                                "top": 30,
                                "left": 20
                            }},
                            "subTopics": [
                                {{
                                    "name": "Sub-topic Name",
                                    "pageNumber": 1,
                                    "position": {{
                                        "top": 40,
                                        "left": 30
                                    }}
                                }}
                            ]
                        }}
                    ]
                }}
            ]
        }}

        Rules:
        1. Return ONLY the JSON object, no other text
        2. Position coordinates should be percentages (0-100)
        3. Page numbers should be integers
        4. Make sure the JSON is properly formatted with double quotes
        5. Do not include any explanations or additional text
        6. Identify clear chapter headings and their associated topics
        7. Include sub-topics where appropriate
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
            analysis = json.loads(response_text)
            
            # Validate the structure
            if 'chapters' not in analysis:
                raise ValueError("Missing chapters in response")
            
            # Validate each chapter
            for chapter in analysis['chapters']:
                if not all(k in chapter for k in ['title', 'pageNumber', 'position', 'topics']):
                    raise ValueError("Invalid chapter structure")
                if not all(k in chapter['position'] for k in ['top', 'left']):
                    raise ValueError("Invalid chapter position structure")
                
                # Validate topics
                for topic in chapter['topics']:
                    if not all(k in topic for k in ['name', 'pageNumber', 'position']):
                        raise ValueError("Invalid topic structure")
                    if not all(k in topic['position'] for k in ['top', 'left']):
                        raise ValueError("Invalid topic position structure")
                    
                    # Validate sub-topics if they exist
                    if 'subTopics' in topic:
                        for sub_topic in topic['subTopics']:
                            if not all(k in sub_topic for k in ['name', 'pageNumber', 'position']):
                                raise ValueError("Invalid sub-topic structure")
                            if not all(k in sub_topic['position'] for k in ['top', 'left']):
                                raise ValueError("Invalid sub-topic position structure")
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse chapter analysis: {str(e)}")
            logging.error(f"Raw response: {response_text}")
            return {
                "chapters": [{
                    "title": "Failed to analyze document",
                    "pageNumber": 1,
                    "position": {
                        "top": 0,
                        "left": 0
                    },
                    "topics": [{
                        "name": "Error occurred",
                        "pageNumber": 1,
                        "position": {
                            "top": 0,
                            "left": 0
                        }
                    }]
                }]
            }
    except Exception as e:
        logging.error(f"Error in chapter analysis: {str(e)}")
        return {
            "chapters": [{
                "title": "Failed to analyze document",
                "pageNumber": 1,
                "position": {
                    "top": 0,
                    "left": 0
                },
                "topics": [{
                    "name": "Error occurred",
                    "pageNumber": 1,
                    "position": {
                        "top": 0,
                        "left": 0
                    }
                }]
            }]
        } 