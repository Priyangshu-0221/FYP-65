"""PDF processing utilities for resume parsing and feature extraction."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from PyPDF2 import PdfReader

from backend.skill_extractor import (
    clean_text,
    extract_skills,
    extract_education_info,
    extract_experience_sentences,
    DEFAULT_SKILLS,
    EDUCATION_KEYWORDS,
    EXPERIENCE_KEYWORDS
)

class PDFProcessor:
    """Handles PDF resume processing and feature extraction."""
    
    def __init__(
        self,
        skills: Optional[List[str]] = None,
        education_keywords: Optional[List[str]] = None,
        experience_keywords: Optional[List[str]] = None
    ):
        """Initialize the PDF processor with optional custom keyword sets.
        
        Args:
            skills: List of skills to search for in resumes
            education_keywords: Keywords to identify education-related content
            experience_keywords: Keywords to identify experience-related content
        """
        self.skills = set(skills or DEFAULT_SKILLS)
        self.education_keywords = set(education_keywords or EDUCATION_KEYWORDS)
        self.experience_keywords = set(experience_keywords or EXPERIENCE_KEYWORDS)
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
        """Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
        """
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = "\n\n".join(page.extract_text() for page in reader.pages)
        return text.strip()
    
    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, str]:
        """Process a single PDF resume and extract structured information.
        
        Args:
            pdf_path: Path to the PDF resume
            
        Returns:
            Dictionary containing extracted information
        """
        # Extract raw text
        raw_text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)
        
        # Extract features
        skills = extract_skills(cleaned_text, self.skills)
        education = extract_education_info(raw_text, self.education_keywords)
        experience = extract_experience_sentences(raw_text, self.experience_keywords)
        
        return {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'skills': skills,
            'education': education,
            'experience': experience,
            'skills_count': len(skills)
        }
    
    def process_pdf_to_dataframe(self, pdf_path: Union[str, Path]) -> pd.DataFrame:
        """Process a single PDF resume and return results as a DataFrame row.
        
        Args:
            pdf_path: Path to the PDF resume
            
        Returns:
            DataFrame with a single row containing the extracted information
        """
        result = self.process_pdf(pdf_path)
        return pd.DataFrame([{
            'filename': Path(pdf_path).name,
            'resume_text': result['raw_text'],
            'skills': result['skills'],
            'education': result['education'],
            'experience': result['experience'],
            'skills_count': result['skills_count']
        }])


def process_pdf_resume(
    pdf_path: Union[str, Path],
    skills: Optional[List[str]] = None,
    education_keywords: Optional[List[str]] = None,
    experience_keywords: Optional[List[str]] = None
) -> Dict[str, str]:
    """Convenience function to process a single PDF resume.
    
    Args:
        pdf_path: Path to the PDF resume
        skills: Optional list of skills to search for
        education_keywords: Optional list of education-related keywords
        experience_keywords: Optional list of experience-related keywords
        
    Returns:
        Dictionary containing extracted information
    """
    processor = PDFProcessor(skills, education_keywords, experience_keywords)
    return processor.process_pdf(pdf_path)
