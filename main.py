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
import re
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar

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
    """Extract main topics from text using Gensim LDA"""
    # Preprocess text
    def preprocess(text):
        # Tokenize and clean text
        tokens = simple_preprocess(text, deacc=True)
        # Remove stopwords and short words
        return [token for token in tokens if token not in STOPWORDS and len(token) > 3]
    
    # Prepare documents
    doc = preprocess(text)
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary([doc])
    corpus = [dictionary.doc2bow(doc)]
    
    # Train LDA model
    num_topics = 10  # Number of topics to extract
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto'
    )
    
    # Extract topics with their importance scores
    topics = []
    for topic_id, topic_terms in lda_model.print_topics():
        # Get all terms for this topic
        terms = topic_terms.split('+')
        # Extract terms and their weights
        for term in terms:
            weight = float(term.split('*')[0].strip())
            term_text = term.split('*')[1].strip().strip('"')
            if weight > 0.1:  # Only include terms with significant weight
                topics.append(Topic(
                    name=term_text,
                    confidence=weight,
                    page_number=page_number,
                    line_number=line_number
                ))
    
    # Sort topics by confidence
    topics.sort(key=lambda x: x.confidence, reverse=True)
    return topics

def clean_text(text: str) -> str:
    """Clean text by removing artifacts and normalizing spaces."""
    # Remove repeated words (common PDF artifact)
    words = text.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i > 0 and word == words[i-1]:
            continue
        cleaned_words.append(word)
    
    # Join words and normalize spaces
    cleaned_text = ' '.join(cleaned_words)
    
    # Remove multiple spaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text

def extract_chapters_with_pdfminer(pdf_path: str) -> List[Chapter]:
    chapters = []

    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        line_font_sizes = []

        # Gather text lines with average font size
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for line_number, text_line in enumerate(element):
                    if isinstance(text_line, LTTextLine):
                        text = text_line.get_text().strip()
                        font_sizes = [char.size for char in text_line if isinstance(char, LTChar)]

                        if text and font_sizes:
                            avg_font_size = sum(font_sizes) / len(font_sizes)
                            line_font_sizes.append({
                                'text': text,
                                'font_size': avg_font_size,
                                'line_number': line_number
                            })

        # Get the max font size and lines containing "chapter"
        if line_font_sizes:
            max_font_size = max(item['font_size'] for item in line_font_sizes)
            candidate_lines = [
                line for line in line_font_sizes
                if abs(line['font_size'] - max_font_size) < 0.5 and 'chapter' in line['text'].lower()
            ]

            chapter_pattern = re.compile(r'(?i)chapter\s*(\d+)[\.\:\-\s]*(.*)')

            for line in candidate_lines:
                match = chapter_pattern.match(line['text'])
                if match:
                    try:
                        chapter_num = int(match.group(1))
                        chapter_title = match.group(2).strip()

                        # Basic title cleanup
                        if '.' in chapter_title:
                            chapter_title = chapter_title.split('.')[0].strip()
                        if '\n' in chapter_title:
                            chapter_title = chapter_title.split('\n')[0].strip()
                        if len(chapter_title.split()) > 10:
                            continue  # too long to be a title

                        chapter_name = f"Chapter {chapter_num}"
                        if chapter_title:
                            chapter_name += f": {chapter_title}"

                        chapters.append(Chapter(
                            number=chapter_num,
                            name=chapter_name,
                            confidence=0.9,
                            page_number=page_num,
                            line_number=line['line_number']
                        ))

                    except ValueError:
                        continue

    return chapters

def parse_line_number(line_number: str) -> int:
    """Parse line number from string, handling various formats."""
    try:
        # Remove markdown formatting and whitespace
        cleaned = line_number.replace('*', '').replace('\n', '').strip()
        
        # Handle range format (e.g., "2,3" or "2-3")
        if ',' in cleaned:
            return int(cleaned.split(',')[0])
        elif '-' in cleaned:
            return int(cleaned.split('-')[0])
        
        # Handle single number
        return int(cleaned)
    except (ValueError, IndexError):
        return 0

def parse_question_options(options_str: str, question_type: str) -> List[str]:
    """Parse options based on question type"""
    if question_type == 'multiple_choice':
        return options_str.strip().split('\n')
    elif question_type == 'true_false':
        return ['True', 'False']
    elif question_type == 'fill_blank':
        return []  # Fill in the blank questions don't have options
    elif question_type == 'short_answer':
        return []  # Short answer questions don't have options
    return []

def find_best_matching_topics(question_text: str, topics: List[Topic], max_topics: int = 3) -> str:
    """Find the best matching topics for a question using text similarity."""
    if not topics:
        return ""
    
    # Convert question to lowercase for better matching
    question_lower = question_text.lower()
    question_words = set(question_lower.split())
    
    # Score each topic based on word overlap and confidence
    topic_scores = []
    for topic in topics:
        score = 0
        topic_words = topic.name.lower().split()
        for word in topic_words:
            if word in question_lower:
                score += 1
        # Combine word overlap score with topic confidence
        final_score = (score / len(topic_words)) * topic.confidence if topic_words else 0
        topic_scores.append((topic.name, final_score))
    
    # Sort topics by score and get top matches
    top_topics = sorted(topic_scores, key=lambda x: x[1], reverse=True)[:max_topics]
    
    # Join top topics with their confidence scores
    return ", ".join([f"{topic} ({score:.2f})" for topic, score in top_topics if score > 0])

def find_best_matching_chapter(question_text: str, chapters: List[Chapter]) -> str:
    """Find the best matching chapter for a question using simple text matching."""
    if not chapters:
        return ""
    
    # Convert question to lowercase for better matching
    question_lower = question_text.lower()
    
    # Score each chapter based on word overlap
    chapter_scores = []
    for chapter in chapters:
        score = 0
        chapter_words = chapter.name.lower().split()
        for word in chapter_words:
            if word in question_lower:
                score += 1
        chapter_scores.append((chapter.name, score))
    
    # Return the chapter with the highest score
    best_chapter = max(chapter_scores, key=lambda x: x[1])
    return best_chapter[0] if best_chapter[1] > 0 else ""

def find_best_matching_page(question_text: str, pages_text: List[str]) -> int:
    """Find the best matching page number for a question using text similarity."""
    if not pages_text:
        return 1
    
    # Convert question to lowercase for better matching
    question_lower = question_text.lower()
    question_words = set(question_lower.split())
    
    # Score each page based on word overlap
    page_scores = []
    for i, page_text in enumerate(pages_text):
        # Convert page text to lowercase and get unique words
        page_lower = page_text.lower()
        page_words = set(page_lower.split())
        
        # Calculate word overlap score
        common_words = question_words.intersection(page_words)
        score = len(common_words) / len(question_words) if question_words else 0
        
        page_scores.append((i + 1, score))  # i + 1 because pages are 1-indexed
    
    # Return the page with the highest score
    best_page = max(page_scores, key=lambda x: x[1])
    return best_page[0] if best_page[1] > 0 else 1

def find_best_matching_line(question_keyword: str, page_text: str) -> int:
    """Find the best matching line number for a question within a page."""
    if not page_text:
        return 0
        
    # Split page text into lines
    lines = page_text.split('\n')
    
    # Convert question to lowercase for better matching
    question_keyword_lower = question_keyword.lower()
    
    # Score each line based on word overlap
    line_scores = []
    for i, line in enumerate(lines):
        # Convert line to lowercase and get unique words
        line_lower = line.lower()
        
        # Calculate word overlap score
        if question_keyword_lower in line_lower:
            score = 1
        else:
            score = 0
        
        line_scores.append((i + 1, score))  # i + 1 because lines are 1-indexed
    
    # Return the line with the highest score
    best_line = max(line_scores, key=lambda x: x[1])
    return best_line[0] if best_line[1] > 0 else 0

def generate_quiz_questions(page_text: str, chapters: List[Chapter] = None, all_pages_text: List[str] = None) -> List[QuizQuestion]:
    """Generate quiz questions for a single page using Gemini."""
    try:
        # Create a prompt that asks for both topics and questions
        prompt = f"""Analyze the following text and:
1. First generate quiz questions as much as possible from the following text. Each question should be on a new line and add page number to the question and with most importance keyword of one line.
2. Then identify the 2 ~ 3 most important topics for the every quiz question
3. After that get Chapter number and title for the every quiz question, Extract Chapter number and title from page text. Here are examples: Chapter 1: Machine Learning, Chapter 2: Deep Learning, Chapter 3: Neural Networks etc.
Text:
{page_text}

First, list the topics in this format:
Topics:
Topic 1
Topic 2
Topic 3

Then, generate quiz questions following this exact format:

For Multiple Choice Questions:
MCQ: [Question text]
Topics: [List of relevant topics from above]
Options:
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
Correct: [A/B/C/D]
Explanation: [Brief explanation]
Page: [Page number]
Chapter: [Chapter number: Chapter title]
Keyword: [Most important keyword of one line]

For True/False Questions:
TF: [Question text]
Topics: [List of relevant topics from above]
Correct: [True/False]
Explanation: [Brief explanation]
Page: [Page number]
Chapter: [Chapter number: Chapter title]
Keyword: [Most important keyword of one line]

For Fill in the Blank Questions:
FIB: [Question text with _____ for blank]
Topics: [List of relevant topics from above]
Answer: [Correct answer]
Explanation: [Brief explanation]
Page: [Page number]
Chapter: [Chapter number: Chapter title]
Keyword: [Most important keyword of one line]

For Short Answer Questions:
SA: [Question text]
Topics: [List of relevant topics from above]
Answer: [Expected answer]
Explanation: [Brief explanation]
Page: [Page number]
Chapter: [Chapter number: Chapter title]
Keyword: [Most important keyword of one line]

Remember:
- Focus on the key topics you identified
- Line numbers should be simple numbers (e.g., "1" or "2-3")
- Each question must be on a new line
- Include all required fields for each question type
- Provide clear and concise explanations
- Make sure questions are relevant to the text and topics
- Include a mix of different question types
- For MCQs, always provide exactly 4 options (A, B, C, D)
- Questions should test understanding of the key topics
- Ensure questions are specific to the content on this page
- Add page number to the question
- Add most important keyword of the question and add it to the question
"""

        # Call Gemini API
        response = generate_with_gemini(prompt)
            
        # Parse the response
        questions = []
        current_question = None
        current_type = None
        collecting_options = False
        page_topics = []
        parsing_topics = True
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Parse topics section
            if parsing_topics:
                if line.startswith('Topics:'):
                    continue
                elif line.startswith('Topic'):
                    topic_name = line.split('Topic')[1].strip()
                    page_topics.append(Topic(
                        name=topic_name,
                        confidence=1.0,  # Since we're not using confidence scores anymore
                        page_number=0,  # Will be updated when we get the page number
                        line_number=0
                    ))
                elif line.startswith('MCQ:') or line.startswith('TF:') or line.startswith('FIB:') or line.startswith('SA:'):
                    parsing_topics = False
                    # Continue with question parsing
                
            # Parse questions section
            if not parsing_topics:
                if line.startswith('MCQ:'):
                    if current_question:
                        questions.append(current_question)
                    current_type = 'multiple_choice'
                    current_question = {
                        'type': current_type,
                        'question': line[4:].strip(),
                        'options': [],
                        'correct_answer': '',
                        'explanation': '',
                        'page_number': 0,  # Will be updated when we get the page number
                        'line_number': 0,
                        'chapter': '',
                        'topic': '',
                        'keyword': ''
                    }
                    collecting_options = False
                elif line.startswith('TF:'):
                    if current_question:
                        questions.append(current_question)
                    current_type = 'true_false'
                    current_question = {
                        'type': current_type,
                        'question': line[3:].strip(),
                        'options': ['True', 'False'],
                        'correct_answer': '',
                        'explanation': '',
                        'page_number': 0,  # Will be updated when we get the page number
                        'line_number': 0,
                        'chapter': '',
                        'topic': '',
                        'keyword': ''
                    }
                    collecting_options = False
                elif line.startswith('FIB:'):
                    if current_question:
                        questions.append(current_question)
                    current_type = 'fill_blank'
                    current_question = {
                        'type': current_type,
                        'question': line[4:].strip(),
                        'options': [],
                        'correct_answer': '',
                        'explanation': '',
                        'page_number': 0,  # Will be updated when we get the page number
                        'line_number': 0,
                        'chapter': '',
                        'topic': '',
                        'keyword': ''
                    }
                    collecting_options = False
                elif line.startswith('SA:'):
                    if current_question:
                        questions.append(current_question)
                    current_type = 'short_answer'
                    current_question = {
                        'type': current_type,
                        'question': line[3:].strip(),
                        'options': [],
                        'correct_answer': '',
                        'explanation': '',
                        'page_number': 0,  # Will be updated when we get the page number
                        'line_number': 0,
                        'chapter': '',
                        'topic': '',
                        'keyword': ''
                    }
                    collecting_options = False
                elif current_question:
                    if line.startswith('Topics:'):
                        # Extract topics for this question
                        topics_str = line[7:].strip()
                        current_question['topic'] = topics_str
                    elif line.startswith('Options:'):
                        collecting_options = True
                        continue
                    elif collecting_options and (line.startswith('A)') or line.startswith('B)') or line.startswith('C)') or line.startswith('D)')):
                        current_question['options'].append(line[3:].strip())
                    elif line.startswith('Correct:'):
                        current_question['correct_answer'] = line[8:].strip()
                        collecting_options = False
                    elif line.startswith('Answer:'):
                        current_question['correct_answer'] = line[7:].strip()
                        collecting_options = False
                    elif line.startswith('Explanation:'):
                        current_question['explanation'] = line[12:].strip()
                        collecting_options = False
                    elif line.startswith('Chapter:'):
                        chapter_str = line[8:].strip()
                        chapter_parts = chapter_str.split(' ', 1)
                        if len(chapter_parts) == 2:
                            current_question['chapter'] = line[8:].strip()
                        else:
                            current_question['chapter'] = ''

                        collecting_options = False
                    elif line.startswith('Keyword:'):
                        current_question['keyword'] = line[9:].strip()
                    elif line.startswith('Page:'):
                        # Extract page number from the response
                        try:
                            page_num = int(line[5:].strip())
                            if page_num == 0:
                                page_num = find_best_matching_page(current_question['question'], all_pages_text)
                            current_question['page_number'] = page_num
                            
                            # Find the best matching line number for this question
                            if all_pages_text and page_num <= len(all_pages_text):
                                page_text = all_pages_text[page_num - 1]
                                line_num = find_best_matching_line(current_question['keyword'], page_text)
                                current_question['line_number'] = line_num
                        except ValueError:
                            print(f"Warning: Could not parse page number from: {line}")
        
        # Add the last question if exists
        if current_question:
            print("--------------------------------")
            print("current_question: ", current_question)
            print("--------------------------------\n")
            questions.append(current_question)
        
        # Add chapter information to each question
        if chapters:
            for question in questions:
                question['chapter'] = find_best_matching_chapter(question['question'], chapters)
                print("question: ", question)
            
        return questions
        
    except Exception as e:
        print(f"Error generating questions: {str(e)}")
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
                # Clean the text before adding it
                cleaned_text = clean_text(text)
                all_text.append(f"Page {page_num + 1}:\n{cleaned_text}")

            # Combine all text
            combined_text = "\n\n".join(all_text)

            # Extract topics and chapters
            topics = []
            chapters = extract_chapters_with_pdfminer(str(final_path))
            
            for page_num, text in enumerate(all_text):
                # Extract topics
                page_topics = extract_topics(text, page_num + 1, 0)
                topics.extend(page_topics)

            # Generate quiz questions for the entire text
            print("\nGenerating questions for the entire document")
            all_questions = generate_quiz_questions(combined_text, chapters, all_text)
            print("all_questions: ", all_questions)
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

# @app.post("/pdf/analyze-pages")
# async def analyze_pages(pages: List[PageContent]) -> AnalysisResponse:
#     """Analyze pages and generate quiz questions"""
#     all_topics = []
#     all_chapters = []
#     all_questions = []
#     all_pages_text = [page.text for page in pages]  # Store all page texts for matching
    
#     for page in pages:
#         # Extract topics
#         topics = extract_topics(page.text, page.page_number, page.line_numbers[0].start)
#         all_topics.extend(topics)
        
#         # Extract chapters
#         chapters = extract_chapters(page.text, page.page_number)
#         all_chapters.extend(chapters)
        
#         # Generate quiz questions with topics, chapters, and all pages text
#         questions = generate_quiz_questions(page.text, all_chapters, all_pages_text)
#         all_questions.extend(questions)
    
#     # Create response
#     response = AnalysisResponse(
#         topics=all_topics,
#         chapters=all_chapters,
#         questions=all_questions
#     )
    
#     # Save to CSV
#     save_to_csv(response.dict(), f"analysis_{pages[0].page_number}")
    
#     return response

# @app.get("/pdf/analysis/{file_id}")
# async def get_analysis(file_id: str) -> AnalysisResponse:
#     """Get analysis results for a file"""
#     csv_path = UPLOAD_DIR / f"analysis_{file_id}.csv"
#     if not csv_path.exists():
#         raise HTTPException(status_code=404, detail="Analysis not found")
    
#     # Read from CSV and convert to response format
#     df = pd.read_csv(csv_path)
    
#     topics = df[df['type'] == 'topic'].to_dict('records')
#     chapters = df[df['type'] == 'chapter'].to_dict('records')
#     questions = df[df['type'] == 'question'].to_dict('records')
    
#     # Convert options back to list
#     for q in questions:
#         q['options'] = json.loads(q['options'])
    
#     return AnalysisResponse(
#         topics=topics,
#         chapters=chapters,
#         questions=questions
#     )
