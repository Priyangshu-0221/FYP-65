"""Pydantic schemas for internship recommender API."""

from __future__ import annotations

from typing import Dict, List, Optional

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
    skills: List[str] = Field(
        default_factory=list, 
        description="List of skills, ordered by proficiency (most proficient first)"
    )
    top_k: int = Field(
        5, 
        ge=1, 
        le=20, 
        description="Number of recommendations to return"
    )
    marks: Optional[Dict[str, float]] = Field(
        None,
        description="Academic marks in format {'cgpa': 3.5, 'percentage': 85.5}"
    )
    skill_count: Optional[int] = Field(
        None,
        description="Total number of skills (will be set to len(skills) if not provided)"
    )


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
