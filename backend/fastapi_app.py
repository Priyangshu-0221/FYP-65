"""FastAPI application exposing resume upload and internship recommendation endpoints."""

from __future__ import annotations

import io
import json
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .internship_catalog import load_catalog
from .pdf_processor import PDFProcessor, process_pdf_resume
from .recommendation_engine import InternshipRecommendationEngine
# from .resume_classifier import predict_resume_category, get_or_train_model, run_training_pipeline
from .schemas import InternshipSchema, RecommendationRequest, RecommendationResponse, ResumeUploadResponse
from .settings import settings
from .skill_extractor import extract_skills_from_text, DEFAULT_SKILLS, EDUCATION_KEYWORDS, EXPERIENCE_KEYWORDS, clean_text, tokenize_sentences
import pandas as pd
from datetime import datetime

# Excel Logging Helper
def log_to_excel(filename: str, data: dict):
    """Append a row of data to an Excel file."""
    file_path = Path("reports") / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_new = pd.DataFrame([data])
    
    if file_path.exists():
        # Append to existing file
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Load existing data to find the next row
            try:
                reader = pd.read_excel(file_path)
                start_row = len(reader) + 1
                df_new.to_excel(writer, index=False, header=False, startrow=start_row)
            except ValueError:
                # File might be empty or corrupted
                df_new.to_excel(writer, index=False)
    else:
        # Create new file
        df_new.to_excel(file_path, index=False)

app = FastAPI(title="Internship Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATALOG = load_catalog(settings.internship_catalog_path)

# Load or train the model on startup
# Load or train the model on startup - DEPRECATED
print("Loading or training model - SKIPPED")
MODEL = None
VECTORIZER = None

# Initialize recommendation engine
print("Initializing recommendation engine...")
try:
    # Use the same catalog path as the app
    RECOMMENDER = InternshipRecommendationEngine(data_path=settings.internship_catalog_path)
    print("Recommendation engine initialized successfully")
except Exception as e:
    print(f"Error initializing recommendation engine: {e}")
    RECOMMENDER = None


def _extract_text_from_upload(content_type: str, raw_bytes: bytes) -> str:
    """Extract text from uploaded file based on content type."""
    if content_type == 'application/pdf':
        # This is now handled by the PDF processor
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(raw_bytes)
            temp_pdf_path = temp_pdf.name
        
        try:
            result = process_pdf_resume(temp_pdf_path)
            return result['raw_text']
        finally:
            # Clean up the temporary file
            Path(temp_pdf_path).unlink(missing_ok=True)
    elif content_type.startswith('text/'):
        return raw_bytes.decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {content_type}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to provide detailed error messages."""
    import traceback
    error_details = {
        "error": str(exc),
        "type": exc.__class__.__name__,
        "traceback": traceback.format_exc(),
    }
    print("\n=== ERROR DETAILS ===")
    print(json.dumps(error_details, indent=2))
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An error occurred while processing your request",
            "error": str(exc),
            "type": exc.__class__.__name__
        }
    )

@app.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    process_as_pdf: str = Form("false"),
):
    """Process an uploaded resume file (PDF or text) and return extracted information."""
    print(f"\n=== New Upload Request ===")
    print(f"Filename: {file.filename}")
    print(f"Content-Type: {file.content_type}")
    print(f"Process as PDF: {process_as_pdf}")
    
    # Initialize response with default values
    response_data = {
        "success": False,
        "text": "",
        "skills": [],
        "category": "",
        "education": "",
        "experience": "",
        "message": ""
    }
    
    is_pdf = process_as_pdf.lower() == "true" or (file.filename and file.filename.lower().endswith('.pdf'))
    
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
            
        print(f"Received file: {file.filename}, size: {len(content)} bytes, is_pdf: {is_pdf}")
        
        # Log first few bytes to verify file content
        print(f"File content starts with: {content[:100]}")
        
        if is_pdf:
            print("Processing as PDF file")
            # Process as PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(content)
                temp_pdf_path = temp_pdf.name
            
            try:
                print(f"Saved PDF to temporary file: {temp_pdf_path}")
                
                # Process the PDF to extract text and basic information
                result = process_pdf_resume(
                    temp_pdf_path, 
                    skills=DEFAULT_SKILLS, 
                    education_keywords=EDUCATION_KEYWORDS, 
                    experience_keywords=EXPERIENCE_KEYWORDS
                )
                
                # Use the global model to predict category
                category = "General"
                # Classifier removed in favor of semantic matching
                # if MODEL and VECTORIZER:
                #     try:
                #         category = predict_resume_category(MODEL, VECTORIZER, result.get('raw_text', ''))
                #         print(f"Predicted category: {category}")
                #     except Exception as model_error:
                #         print(f"Warning: Could not predict category: {str(model_error)}")

                
                # Log to Excel
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                raw_text = result.get('raw_text', '')
                cleaned = result.get('cleaned_text', '') or clean_text(raw_text)
                
                # 1. Raw Data
                log_to_excel("raw_data.xlsx", {
                    "Timestamp": timestamp,
                    "Filename": file.filename,
                    "Raw_Content": raw_text
                })
                
                # 2. Cleaned Data
                log_to_excel("cleaned_data.xlsx", {
                    "Timestamp": timestamp,
                    "Filename": file.filename,
                    "Cleaned_Content": cleaned
                })
                
                # 3. Tokenized Data
                # Simple whitespace tokenization for the report
                tokens = cleaned.split()
                log_to_excel("tokenized_data.xlsx", {
                    "Timestamp": timestamp,
                    "Filename": file.filename,
                    "Tokens": str(tokens)
                })

                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "success": True,
                        "text": raw_text[:1000] + '...' if len(raw_text) > 1000 else raw_text,
                        "skills": result.get('skills', []),
                        "category": category,
                        "education": result.get('education', ''),
                        "experience": result.get('experience', '')
                    }
                )
                
            except Exception as e:
                import traceback
                error_msg = f"Error processing PDF: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "success": False, 
                        "message": "Error processing resume. Please try again.",
                        "error": str(e)
                    }
                )
                
            finally:
                # Clean up the temporary file
                try:
                    Path(temp_pdf_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to delete temp file {temp_pdf_path}: {str(e)}")
        
        else:
            print("Processing as text file")
            # Process as plain text
            resume_text = _extract_text_from_upload(file.content_type, content)
            if not resume_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No text content found in the uploaded file"
                )
                
            print(f"Extracted {len(resume_text)} characters of text")
            skills = extract_skills_from_text(resume_text)
            print(f"Extracted {len(skills)} skills from text")
            
            return ResumeUploadResponse(
                success=True,
                text=resume_text[:500] + '...' if len(resume_text) > 500 else resume_text,
                skills=skills,
                category="General", # Classifier removed
                message="Text resume processed successfully"
            )
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        error_msg = f"Unexpected error processing file: {str(e)}"
        print(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


# Endpoint removed as classifier is deprecated


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_internships(payload: RecommendationRequest) -> RecommendationResponse:
    """
    Get internship recommendations based on the provided skills, marks, and skill count.
    
    The recommendation engine considers:
    - Skill matches (weighted by position in the skills list)
    - Academic marks (if provided)
    - Total number of skills (if provided)
    """
    if not payload.skills:
        raise HTTPException(status_code=400, detail="No skills provided for recommendation.")

    try:
        if not RECOMMENDER:
            raise HTTPException(status_code=500, detail="Recommendation engine not initialized")
        
        # Get recommendations with marks and skill count if provided
        recommendations = RECOMMENDER.recommend_internships(
            user_skills=payload.skills,
            user_marks=payload.marks if hasattr(payload, 'marks') else None,
            skill_count=len(payload.skills),
            top_n=payload.top_k
        )
        
        # Convert to the expected response model
        return RecommendationResponse(recommendations=[
            InternshipSchema(
                id=item['id'],
                title=item['title'],
                company=item['company'],
                location=item['location'],
                category=item.get('domain', 'General'),
                skills=[s['name'] if isinstance(s, dict) else s for s in item['skills_required']],
                description=item.get('description', ''),
                apply_link=item.get('apply_link', '#')
            ) for item in recommendations
        ])
        
    except Exception as e:
        error_msg = f"Error generating recommendations: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
