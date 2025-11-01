"""
PDF processing utilities for resume parsing and feature extraction.

This module provides functionality to process PDF resumes, extract text, and
identify key information such as skills, education, work experience, and marks.
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import Counter

import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm

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
    
    def process_resume(self, pdf_path: Union[str, Path], category: str = None) -> Dict[str, Any]:
        """Process a single resume PDF and extract key features.
        
        Args:
            pdf_path: Path to the PDF resume
            category: Optional category of the resume
            
        Returns:
            Dictionary containing extracted resume information
        """
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                return None
            
            # Clean and process text
            cleaned_text = clean_text(text)
            
            # Extract features
            skills = extract_skills(cleaned_text, self.skills)
            education = extract_education_info(cleaned_text, self.education_keywords)
            experience = extract_experience_sentences(cleaned_text, self.experience_keywords)
            
            # Extract marks from education section
            education_text = '\n'.join(education)
            marks = self.extract_marks(education_text)
            
            # Extract skills with frequency
            skills_with_freq = self.extract_skills_with_frequency(cleaned_text, list(self.skills))
            
            # Prepare result
            result = {
                'file_path': str(pdf_path),
                'category': category,
                'text': text[:1000] + '...' if len(text) > 1000 else text,  # Store first 1000 chars
                'skills': list(skills),
                'skills_with_frequency': skills_with_freq,
                'education': education,
                'experience': experience,
                **marks  # Add marks (cgpa, percentage) to the result
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    def process_resume_directory(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        """Process all resumes in the given directory and its subdirectories.
        
        Args:
            data_dir: Directory containing resume PDFs, organized in subdirectories by category
            
        Returns:
            DataFrame with extracted resume information
        """
        data_dir = Path(data_dir)
        if not data_dir.exists() or not data_dir.is_dir():
            raise ValueError(f"Directory not found: {data_dir}")
        
        results = []
        
        # Get all PDF files in the directory tree
        pdf_files = list(data_dir.glob('**/*.pdf'))
        
        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            return pd.DataFrame()
        
        # Process each PDF file
        for pdf_file in tqdm(pdf_files, desc="Processing resumes"):
            # Get category from parent directory name
            category = pdf_file.parent.name
            
            # Process the resume
            result = self.process_resume(pdf_file, category)
            if result:
                results.append(result)
        
        # Convert to DataFrame
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    @staticmethod
    def extract_marks(education_text: str) -> Dict[str, float]:
        """Extract marks/CGPA from education section text.
        
        Args:
            education_text: Text containing education information
        
        Returns:
            Dictionary with extracted marks (e.g., {'cgpa': 3.5, 'percentage': 85.0})
        """
        marks = {}
        
        # Look for CGPA patterns (e.g., CGPA: 3.5/4.0 or 8.5/10)
        cgpa_patterns = [
            r'CGPA[\s:]*([\d\.]+)\s*/\s*[\d\.]+',
            r'GPA[\s:]*([\d\.]+)\s*/\s*[\d\.]+',
            r'([\d\.]+)\s*/\s*[\d\.]+\s*CGPA',
            r'([\d\.]+)\s*/\s*[\d\.]+'
        ]
        
        for pattern in cgpa_patterns:
            matches = re.findall(pattern, education_text, re.IGNORECASE)
            if matches:
                try:
                    marks['cgpa'] = float(matches[0])
                    break
                except (ValueError, IndexError):
                    continue
        
        # Look for percentage patterns
        percent_patterns = [
            r'(?:percentage|%|percent|pct)[\s:]*([\d\.]+)',
            r'([\d\.]+)\s*%',
            r'([\d\.]+)\s*(?:percent|pct)'
        ]
        
        for pattern in percent_patterns:
            matches = re.findall(pattern, education_text, re.IGNORECASE)
            if matches:
                try:
                    marks['percentage'] = float(matches[0])
                    break
                except (ValueError, IndexError):
                    continue
        
        return marks
    
    @staticmethod
    def extract_skills_with_frequency(text: str, skills_list: List[str]) -> Dict[str, int]:
        """Extract skills and their frequency from text.
        
        Args:
            text: Input text to search for skills
            skills_list: List of skills to search for
        
        Returns:
            Dictionary with skills as keys and their frequency as values
        """
        text = clean_text(text)
        tokens = text.lower().split()
        skill_counts = {}
        
        for skill in skills_list:
            skill_lower = skill.lower()
            # Count exact matches (handles multi-word skills)
            count = text.lower().count(skill_lower)
            if count > 0:
                skill_counts[skill] = count
        
        return skill_counts


def process_pdf_resume(
    pdf_path: Union[str, Path],
    skills: Optional[List[str]] = None,
    education_keywords: Optional[List[str]] = None,
    experience_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
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
    return processor.process_resume(pdf_path)


def process_resumes(data_dir: Union[str, Path], output_file: str = None) -> pd.DataFrame:
    """Process all resumes in a directory and optionally save results to a CSV file.
    
    Args:
        data_dir: Directory containing resume PDFs, organized in subdirectories by category
        output_file: Optional path to save the results as CSV
        
    Returns:
        DataFrame with extracted resume information
    """
    processor = PDFProcessor()
    df = processor.process_resume_directory(data_dir)
    
    if output_file and not df.empty:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed {len(df)} resumes. Results saved to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDF resumes and extract features")
    parser.add_argument("input_dir", type=str, help="Directory containing resume PDFs")
    parser.add_argument("-o", "--output", type=str, default="processed_resumes.csv",
                       help="Output CSV file path (default: processed_resumes.csv)")
    
    args = parser.parse_args()
    
    print(f"Processing resumes in {args.input_dir}...")
    df = process_resumes(args.input_dir, args.output)
    
    if df.empty:
        print("No valid resumes found or processed.")
    else:
        print(f"Successfully processed {len(df)} resumes.")
