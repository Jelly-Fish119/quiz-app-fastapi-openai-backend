from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  
from typing import List
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import os
import tempfile
import shutil
from pathlib import Path 
import csv
import json
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

app = FastAPI(
    title="PDF Topic Extractor",
    version="1.0"
)

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_with_gemini(prompt: str, max_retries: int = 3) -> str:
    """Generate text using Gemini API with retries"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating content after {max_retries} attempts: {str(e)}"
                )

# Data models
class LineNumber(BaseModel):
    start: int
    end: int

class PageContent(BaseModel):
    page_number: int
    text: str
    line_numbers: List[LineNumber]

class Topic(BaseModel):
    name: str
    confidence: float
    page_number: int
    line_number: int

class Chapter(BaseModel):
    number: int
    name: str
    confidence: float
    page_number: int
    line_number: int

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    type: str
    page_number: int
    line_number: int
    chapter: str
    topic: str

class AnalysisResponse(BaseModel):
    topics: List[Topic]
    chapters: List[Chapter]
    questions: List[QuizQuestion]

class UploadForm(BaseModel):
    file_name: str
    total_chunks: int

# File storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
CHUNKS_DIR = UPLOAD_DIR / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

def save_to_csv(data: dict, filename: str):
    """Save analysis data to CSV file"""
    csv_path = UPLOAD_DIR / f"{filename}.csv"
    
    # Define all possible fields
    fieldnames = [
        'type',
        'name',
        'confidence',
        'page_number',
        'line_number',
        'question',
        'options',
        'correct_answer',
        'explanation',
        'question_type',
        'chapter',
        'topic'
    ]
    
    # Convert data to flat structure for CSV
    rows = []
    for topic in data['topics']:
        rows.append({
            'type': 'topic',
            'name': topic['name'],
            'confidence': topic['confidence'],
            'page_number': topic['page_number'],
            'line_number': topic['line_number'],
            'question': '',
            'options': '',
            'correct_answer': '',
            'explanation': '',
            'question_type': '',
            'chapter': '',
            'topic': ''
        })
    
    for chapter in data['chapters']:
        rows.append({
            'type': 'chapter',
            'name': chapter['name'],
            'confidence': chapter['confidence'],
            'page_number': chapter['page_number'],
            'line_number': chapter['line_number'],
            'question': '',
            'options': '',
            'correct_answer': '',
            'explanation': '',
            'question_type': '',
            'chapter': '',
            'topic': ''
        })
    
    for question in data['questions']:
        rows.append({
            'type': 'question',
            'name': '',
            'confidence': '',
            'page_number': question['page_number'],
            'line_number': question['line_number'],
            'question': question['question'],
            'options': json.dumps(question['options']),
            'correct_answer': question['correct_answer'],
            'explanation': question['explanation'],
            'question_type': question['type'],
            'chapter': question['chapter'],
            'topic': question['topic']
        })
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def extract_topics(text: str, page_number: int, line_number: int) -> List[Topic]:
    """Extract main topics from text using NLTK"""
    # Tokenize and clean text
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Get frequency distribution
    fdist = FreqDist(filtered_tokens)
    
    # Extract top topics
    topics = []
    for word, freq in fdist.most_common(5):
        confidence = freq / len(filtered_tokens)
        topics.append(Topic(
            name=word,
            confidence=confidence,
            page_number=page_number,
            line_number=line_number
        ))
    
    return topics

def extract_chapters(text: str, page_number: int) -> List[Chapter]:
    """Extract chapter information from text"""
    # Look for chapter patterns
    sentences = sent_tokenize(text)
    chapters = []
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in ['chapter', 'section', 'part']):
            # Try to extract chapter number
            import re
            number_match = re.search(r'(?:chapter|section|part)\s*(\d+)', sentence_lower)
            if number_match:
                chapter_number = int(number_match.group(1))
                chapters.append(Chapter(
                    number=chapter_number,
                    name=sentence.strip(),
                    confidence=0.9,  # High confidence for explicit chapter markers
                    page_number=page_number,
                    line_number=i
                ))
    
    return chapters

def parse_line_number(line_str: str) -> int:
    """Parse line number, handling both single numbers and ranges"""
    try:
        # If it's a range (e.g., "4-5"), take the first number
        if '-' in line_str:
            return int(line_str.split('-')[0])
        # Otherwise, try to parse as a single number
        return int(line_str)
    except (ValueError, IndexError):
        # If parsing fails, return 0 as default
        return 0

def generate_quiz_questions(text: str, page_number: int = 1) -> List[QuizQuestion]:
    """Generate quiz questions using Gemini for the entire text"""
    try:
        # Create a comprehensive prompt for the entire text
        prompt = f"""Create 10-15 high-quality quiz questions for this educational content:

{text[:4000]}  # Limit text to first 4000 characters to avoid token limits

Format each question as:
Question: [clear, specific question about the content]
Options: [A]
[B]
[C]
[D]
Correct Answer: [option]
Explanation: [detailed explanation of why this is correct and why others are incorrect]
Type: multiple_choice
Page: [page number where the answer can be found]
Line: [line number or range, e.g., 4-5]
Chapter: [chapter name or number if applicable]
Topic: [main topic this question covers]

Requirements:
- Questions should cover key concepts from the content
- Make options plausible and well-distributed
- Provide detailed explanations that help with learning
- Include specific references to the content in explanations
- Vary question difficulty
- Cover different aspects of the content

Separate questions with blank lines."""
        
        # Generate questions for the entire text
        response = generate_with_gemini(prompt)
        print("Generated questions for the content")
        
        # Parse response and create QuizQuestion objects
        all_questions = []
        for q in response.split('\n\n'):
            if 'Question:' in q:
                try:
                    question_data = {
                        'question': q.split('Question:')[1].split('Options:')[0].strip(),
                        'options': q.split('Options:')[1].split('Correct Answer:')[0].strip().split('\n'),
                        'correct_answer': q.split('Correct Answer:')[1].split('Explanation:')[0].strip(),
                        'explanation': q.split('Explanation:')[1].split('Type:')[0].strip(),
                        'type': q.split('Type:')[1].split('Page:')[0].strip(),
                        'page_number': int(q.split('Page:')[1].split('Line:')[0].strip()),
                        'line_number': parse_line_number(q.split('Line:')[1].split('Chapter:')[0].strip()),
                        'chapter': q.split('Chapter:')[1].split('Topic:')[0].strip(),
                        'topic': q.split('Topic:')[1].strip()
                    }
                    all_questions.append(QuizQuestion(**question_data))
                except Exception as e:
                    print(f"Error parsing question: {e}")
                    continue
        
        print(f"Total questions generated: {len(all_questions)}")
        return all_questions
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

@app.post("/pdf/upload-chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    file_name: str = Form(...)
):
    """Handle chunked file upload"""
    chunk_dir = CHUNKS_DIR / file_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_path = chunk_dir / f"chunk_{chunk_index}"
    print(f"Uploading chunk {chunk_index} of {total_chunks} to {chunk_path}")
    
    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"Chunk {chunk_index} uploaded successfully to {chunk_path}")
    return {"message": "Chunk uploaded successfully"}

@app.post("/pdf/finalize-upload")
async def finalize_upload(
    file_name: str = Form(...),
    total_chunks: int = Form(...)
):
    """Combine chunks and start processing"""
    chunk_dir = CHUNKS_DIR / file_name
    print(f"Looking for chunks in {chunk_dir}")
    
    if not chunk_dir.exists():
        print(f"Chunk directory not found: {chunk_dir}")
        raise HTTPException(status_code=404, detail="Chunks not found")
    
    # Combine chunks
    final_path = UPLOAD_DIR / file_name
    with open(final_path, "wb") as outfile:
        for i in range(total_chunks):
            chunk_path = chunk_dir / f"chunk_{i}"
            print(f"Reading chunk {i} from {chunk_path}")
            if not chunk_path.exists():
                print(f"Chunk {i} not found at {chunk_path}")
                raise HTTPException(status_code=404, detail=f"Chunk {i} not found")
            with open(chunk_path, "rb") as infile:
                outfile.write(infile.read())
    
    # Clean up chunks with retries
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # First try to remove individual files
            for i in range(total_chunks):
                chunk_path = chunk_dir / f"chunk_{i}"
                if chunk_path.exists():
                    chunk_path.unlink()
            
            # Then try to remove the directory
            if chunk_dir.exists():
                chunk_dir.rmdir()
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} to clean up chunks failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
            else:
                print("Warning: Could not clean up chunks directory. Continuing with processing.")

    # Process the PDF
    try:
        import PyPDF2
        with open(final_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            all_text = []
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                all_text.append(f"Page {page_num + 1}:\n{text}")

            # Combine all text
            combined_text = "\n\n".join(all_text)

            # Extract topics and chapters
            topics = []
            chapters = []
            for page_num, text in enumerate(all_text):
                # Extract topics
                page_topics = extract_topics(text, page_num + 1, 0)
                topics.extend(page_topics)
                
                # Extract chapters
                page_chapters = extract_chapters(text, page_num + 1)
                chapters.extend(page_chapters)

            # Generate quiz questions for the entire text
            print("\nGenerating questions for the entire document")
            all_questions = generate_quiz_questions(combined_text)

            # Save analysis results
            analysis = AnalysisResponse(
                topics=topics,
                chapters=chapters,
                questions=all_questions
            )
            
            # Save to CSV
            save_to_csv(analysis.dict(), f"analysis_{file_name}")
            
            return {"fileId": file_name, "analysis": analysis}

    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/pdf/analyze-pages")
async def analyze_pages(pages: List[PageContent]) -> AnalysisResponse:
    """Analyze pages and generate quiz questions"""
    all_topics = []
    all_chapters = []
    all_questions = []
    
    for page in pages:
        # Extract topics
        topics = extract_topics(page.text, page.page_number, page.line_numbers[0].start)
        all_topics.extend(topics)
        
        # Extract chapters
        chapters = extract_chapters(page.text, page.page_number)
        all_chapters.extend(chapters)
        
        # Generate quiz questions
        questions = generate_quiz_questions(page.text, page.page_number)
        all_questions.extend(questions)
    
    # Create response
    response = AnalysisResponse(
        topics=all_topics,
        chapters=all_chapters,
        questions=all_questions
    )
    
    # Save to CSV
    save_to_csv(response.dict(), f"analysis_{pages[0].page_number}")
    
    return response

@app.get("/pdf/analysis/{file_id}")
async def get_analysis(file_id: str) -> AnalysisResponse:
    """Get analysis results for a file"""
    csv_path = UPLOAD_DIR / f"analysis_{file_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Read from CSV and convert to response format
    df = pd.read_csv(csv_path)
    
    topics = df[df['type'] == 'topic'].to_dict('records')
    chapters = df[df['type'] == 'chapter'].to_dict('records')
    questions = df[df['type'] == 'question'].to_dict('records')
    
    # Convert options back to list
    for q in questions:
        q['options'] = json.loads(q['options'])
    
    return AnalysisResponse(
        topics=topics,
        chapters=chapters,
        questions=questions
    )
