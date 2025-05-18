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
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar, LTPage

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
    keyword: str

class AnalysisResponse(BaseModel):
    topics: List[Topic]
    questions: List[QuizQuestion]

class UploadForm(BaseModel):
    file_name: str
    total_chunks: int

# File storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
CHUNKS_DIR = UPLOAD_DIR / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

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

def find_best_matching_line(question_keyword: str, page_object: LTPage) -> int:
    """Find the best matching line number for a question within a page."""
    if not page_object:
        return 0
        
    temp_number = -1
    for element in page_object:
        if isinstance(element, LTTextContainer):
            for text_line in element:
                if isinstance(text_line, LTTextLine):
                    line_text = text_line.get_text().strip()
                    temp_number += 1
                    if question_keyword.lower() in line_text.lower():
                        print("In 274, find_best_matching_line, temp_number: ", temp_number, "and line_text: ", line_text, "\n")
                        return temp_number
    

def generate_quiz_questions(page_text: str, all_pages_text: List[str] = None, pdf_page_objects: List[LTPage] = None) -> List[QuizQuestion]:
    """Generate quiz questions for a single page using Gemini."""
    try:
        # Create a prompt that asks for both topics and questions
        prompt = f"""Analyze the following text and:
1. First generate quiz questions as much as possible from the following text. Each question should be on a new line and add page number to the question and with a important short keyword combined words in the line of the page.
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
- Add most important keyword of the question and add it to the question, don't ruin the word order and pattern in the sentences
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
                                line_num = find_best_matching_line(current_question['keyword'], pdf_page_objects[page_num - 1])
                                current_question['line_number'] = line_num
                        except ValueError:
                            print(f"Warning: Could not parse page number from: {line}")
        
        # Add the last question if exists
        if current_question:
            print("--------------------------------")
            print("current_question: ", current_question)
            print("--------------------------------\n")
            questions.append(current_question)
        print("In 688, generate_quiz_questions, questions: ", questions, "\n")
        print("--------------------------------\n")
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
            pdf_page_objects = list(extract_pages(file))
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

            # Extract topics
            topics = []
            
            for page_num, text in enumerate(all_text):
                # Extract topics
                page_topics = extract_topics(text, page_num + 1, 0)
                topics.extend(page_topics)

            # Generate quiz questions for the entire text
            print("\nGenerating questions for the entire document")
            all_questions = generate_quiz_questions(combined_text, all_text, pdf_page_objects)
            print("all_questions: ", all_questions)
            # Save analysis results
            analysis = AnalysisResponse(
                topics=topics,
                questions=all_questions
            )
                        
            return {"fileId": file_name, "analysis": analysis}

    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
