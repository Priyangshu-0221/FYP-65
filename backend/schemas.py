"""Pydantic schemas for internship recommender API."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ResumeUploadResponse(BaseModel):
    success: bool = Field(True, description="Indicates if the request was successful")
    text: str = Field("", description="Extracted text from the resume (truncated if too long)")
    skills: List[str] = Field(default_factory=list, description="Extracted skills from resume")
    category: str = Field("", description="Predicted category from classifier")
    education: str = Field("", description="Extracted education information")
    experience: str = Field("", description="Extracted experience information")
    message: str = Field("", description="Status message about the processing")


class RecommendationRequest(BaseModel):
    skills: List[str] = Field(default_factory=list, description="Skills extracted from the resume")
    top_k: int = Field(5, ge=1, le=20, description="Number of recommendations to return")


class InternshipSchema(BaseModel):
    id: str
    title: str
    company: str
    location: str
    category: str
    skills: List[str]
    description: str = ""
    apply_link: str = ""


class RecommendationResponse(BaseModel):
    recommendations: List[InternshipSchema]
