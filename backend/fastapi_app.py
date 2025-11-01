"""FastAPI application exposing resume upload and internship recommendation endpoints."""

from __future__ import annotations

import io
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.internship_catalog import load_catalog
from backend.pdf_processor import PDFProcessor, process_pdf_resume
from backend.recommendation_engine import rank_internships
from backend.resume_classifier import predict_resume_category, get_or_train_model
from backend.schemas import InternshipSchema, RecommendationRequest, RecommendationResponse, ResumeUploadResponse
from backend.settings import settings
from backend.skill_extractor import extract_skills_from_text, DEFAULT_SKILLS, EDUCATION_KEYWORDS, EXPERIENCE_KEYWORDS

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
print("Loading or training model...")
try:
    MODEL, VECTORIZER = get_or_train_model(
        data_path=settings.resume_dataset_path,
        model_type="logistic_regression",
        test_size=0.2,
        random_state=42,
        max_features=5000
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading/training model: {e}")
    MODEL = None
    VECTORIZER = None


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
                
                # Try to get a category if possible, otherwise use a default
                category = "General"
                try:
                    # Try to load the model if the data is available
                    data_dir = Path(__file__).parent.parent / "DATA" / "data"
                    if data_dir.exists() and data_dir.is_dir():
                        try:
                            print(f"Loading resume data from: {data_dir}")
                            model, vectorizer = run_training_pipeline(data_dir)
                            category = predict_resume_category(model, vectorizer, result.get('raw_text', ''))
                            print(f"Predicted category: {category}")
                        except Exception as e:
                            print(f"Warning: Could not train model: {str(e)}\n{traceback.format_exc()}")
                            category = "General"
                    else:
                        print(f"Warning: Resume dataset directory not found at {data_dir}. Using default category.")
                        category = "General"
                except Exception as model_error:
                    print(f"Warning: Could not predict category: {str(model_error)}")
                
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={
                        "success": True,
                        "text": result.get('raw_text', '')[:1000] + '...' if len(result.get('raw_text', '')) > 1000 else result.get('raw_text', ''),
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
                category=predict_resume_category(resume_text),
                message="Text resume processed successfully"
            )
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is


@app.post("/api/retrain-model")
async def retrain_model():
    """
    Retrain the model with the latest data.
    This is an admin endpoint that should be protected in production.
    """
    global MODEL, VECTORIZER
    
    try:
        # Force retraining by setting force_retrain=True
        MODEL, VECTORIZER = get_or_train_model(
            data_path=settings.resume_dataset_path,
            model_type="logistic_regression",
            force_retrain=True,
            test_size=0.2,
            random_state=42,
            max_features=5000
        )
        
        if MODEL is None or VECTORIZER is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrain the model"
            )
            
        return {
            "success": True,
            "message": "Model retrained successfully",
            "model_type": MODEL.__class__.__name__
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retraining model: {str(e)}"
        )
        
    except Exception as e:
        error_msg = f"Unexpected error processing file: {str(e)}"
        print(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_internships(payload: RecommendationRequest) -> RecommendationResponse:
    if not payload.skills:
        raise HTTPException(status_code=400, detail="No skills provided for recommendation.")

    ranked = rank_internships(payload.skills, CATALOG, top_k=payload.top_k)
    recommendations = [InternshipSchema(**internship.to_dict()) for internship, _score in ranked]
    return RecommendationResponse(recommendations=recommendations)
