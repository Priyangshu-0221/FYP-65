"""FastAPI application exposing resume upload and internship recommendation endpoints."""

from __future__ import annotations

import io
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader

from resume_classifier import predict_resume_category, run_training_pipeline

from .internship_catalog import load_catalog
from .recommendation_engine import rank_internships
from .schemas import InternshipSchema, RecommendationRequest, RecommendationResponse, ResumeUploadResponse
from .settings import settings
from .skill_extractor import extract_skills_from_text

app = FastAPI(title="Internship Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATALOG = load_catalog(settings.internship_catalog_path)

try:
    _training_results = run_training_pipeline(settings.resume_dataset_path, model_type="logistic_regression")
    MODEL = _training_results["model"]
    VECTORIZER = _training_results["vectorizer"]
except Exception:
    MODEL = None
    VECTORIZER = None


def _extract_text_from_upload(content_type: str, raw_bytes: bytes) -> str:
    if content_type == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return " ".join(pages)
        except Exception as exc:  # pragma: no cover - PyPDF errors vary
            raise HTTPException(status_code=400, detail="Unable to read PDF content") from exc
    if content_type == "text/plain":
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="ignore")
    raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or plain text resumes.")


@app.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...)) -> ResumeUploadResponse:
    raw_bytes = await file.read()
    text = _extract_text_from_upload(file.content_type or "", raw_bytes)

    skills = extract_skills_from_text(text)
    categories: List[str] = []
    if MODEL is not None and VECTORIZER is not None:
        try:
            predicted = predict_resume_category(MODEL, VECTORIZER, text)
            categories.append(predicted)
        except Exception:
            categories = []

    return ResumeUploadResponse(skills=skills, categories=categories)


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_internships(payload: RecommendationRequest) -> RecommendationResponse:
    if not payload.skills:
        raise HTTPException(status_code=400, detail="No skills provided for recommendation.")

    ranked = rank_internships(payload.skills, CATALOG, top_k=payload.top_k)
    recommendations = [InternshipSchema(**internship.to_dict()) for internship, _score in ranked]
    return RecommendationResponse(recommendations=recommendations)
