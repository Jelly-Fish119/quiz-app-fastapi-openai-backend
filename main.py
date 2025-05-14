from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  
from typing import List
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import google.generativeai as genai
import os
import tempfile
import shutil
from pathlib import Path
import csv
import json

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

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')

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
    
    # Convert data to flat structure for CSV
    rows = []
    for topic in data['topics']:
        rows.append({
            'type': 'topic',
            'name': topic['name'],
            'confidence': topic['confidence'],
            'page_number': topic['page_number'],
            'line_number': topic['line_number']
        })
    
    for chapter in data['chapters']:
        rows.append({
            'type': 'chapter',
            'name': chapter['name'],
            'confidence': chapter['confidence'],
            'page_number': chapter['page_number'],
            'line_number': chapter['line_number']
        })
    
    for question in data['questions']:
        rows.append({
            'type': 'question',
            'question': question['question'],
            'options': json.dumps(question['options']),
            'correct_answer': question['correct_answer'],
            'explanation': question['explanation'],
            'question_type': question['type'],
            'page_number': question['page_number'],
            'line_number': question['line_number']
        })
    
    # Write to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
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

def generate_quiz_questions(text: str, page_number: int) -> List[QuizQuestion]:
    """Generate quiz questions using Gemini"""
    prompt = f"""
    Generate quiz questions from the following text. Include multiple choice, true/false, fill in the blanks, and short answer questions.
    For each question, provide:
    1. The question
    2. Options (for multiple choice)
    3. Correct answer
    4. Explanation
    5. Type of question
    
    Text: {text}
    """
    
    response = model.generate_content(prompt)
    questions = []
    
    # Parse Gemini response and create QuizQuestion objects
    # This is a simplified version - you'll need to parse the actual response format
    for q in response.text.split('\n\n'):
        if 'Question:' in q:
            questions.append(QuizQuestion(
                question=q.split('Question:')[1].split('Options:')[0].strip(),
                options=q.split('Options:')[1].split('Correct Answer:')[0].strip().split('\n'),
                correct_answer=q.split('Correct Answer:')[1].split('Explanation:')[0].strip(),
                explanation=q.split('Explanation:')[1].split('Type:')[0].strip(),
                type=q.split('Type:')[1].strip(),
                page_number=page_number,
                line_number=0  # You might want to determine this based on the question content
            ))
    
    return questions

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
    
    # Clean up chunks
    shutil.rmtree(chunk_dir)

    # Process the PDF
    try:
        import PyPDF2
        with open(final_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            all_text = []
            page_texts = []
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                page_texts.append(text)
                all_text.append(f"Page {page_num + 1}:\n{text}")

            # Extract topics and chapters
            topics = []
            chapters = []
            for page_num, text in enumerate(page_texts):
                # Extract topics
                page_topics = extract_topics(text, page_num + 1, 0)
                topics.extend(page_topics)
                
                # Extract chapters
                page_chapters = extract_chapters(text, page_num + 1)
                chapters.extend(page_chapters)

            # Generate quiz questions with references
            prompt = (
                "Generate quiz questions from the following text. For each question, provide:\n"
                "1. The question\n"
                "2. Options (for multiple choice)\n"
                "3. Correct answer\n"
                "4. Explanation\n"
                "5. Type of question\n"
                "6. Page number reference\n"
                "7. Line number reference\n"
                "8. Chapter reference (if applicable)\n"
                "9. Topic reference (if applicable)\n\n"
                f"Text content:\n{chr(10).join(all_text)}\n\n"
                f"Chapters found:\n{json.dumps([{'name': c.name, 'number': c.number, 'page': c.page_number, 'line': c.line_number} for c in chapters], indent=2)}\n\n"
                f"Topics found:\n{json.dumps([{'name': t.name, 'page': t.page_number, 'line': t.line_number} for t in topics], indent=2)}"
            )
            
            response = model.generate_content(prompt)
            questions = []
            
            # Parse Gemini response and create QuizQuestion objects with references
            for q in response.text.split('\n\n'):
                if 'Question:' in q:
                    try:
                        question_data = {
                            'question': q.split('Question:')[1].split('Options:')[0].strip(),
                            'options': q.split('Options:')[1].split('Correct Answer:')[0].strip().split('\n'),
                            'correct_answer': q.split('Correct Answer:')[1].split('Explanation:')[0].strip(),
                            'explanation': q.split('Explanation:')[1].split('Type:')[0].strip(),
                            'type': q.split('Type:')[1].split('Page:')[0].strip(),
                            'page_number': int(q.split('Page:')[1].split('Line:')[0].strip()),
                            'line_number': int(q.split('Line:')[1].split('Chapter:')[0].strip()),
                            'chapter': q.split('Chapter:')[1].split('Topic:')[0].strip(),
                            'topic': q.split('Topic:')[1].strip()
                        }
                        questions.append(QuizQuestion(**question_data))
                    except Exception as e:
                        print(f"Error parsing question: {e}")
                        continue

            # Save analysis results
            analysis = AnalysisResponse(
                topics=topics,
                chapters=chapters,
                questions=questions
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
