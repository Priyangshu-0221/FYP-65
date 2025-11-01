"""
Process resume PDFs and extract key features including skills, education, and marks.

This script processes PDF resumes to extract:
- Skills and their frequencies
- Education information
- Marks/GPAs
- Other relevant features for analysis
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm

from pdf_processor import PDFProcessor, process_pdf_resume
from skill_extractor import (
    DEFAULT_SKILLS,
    EDUCATION_KEYWORDS,
    extract_education_info,
    clean_text
)

def extract_marks(education_text: str) -> Dict[str, float]:
    """Extract marks/CGPA from education section."""
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
        r'([\d\.]+)\s*(?:percent|pct|%)'
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

def extract_skills_with_frequency(text: str, skills_list: List[str]) -> Tuple[Dict[str, int], int]:
    """Extract skills and their frequency from text."""
    text_lower = text.lower()
    skills_found = {}
    
    # Check for each skill
    for skill in skills_list:
        skill_lower = skill.lower()
        # Count occurrences of each skill
        count = text_lower.count(skill_lower)
        if count > 0:
            skills_found[skill] = count
    
    # Sort skills by frequency (descending)
    sorted_skills = dict(sorted(skills_found.items(), key=lambda x: x[1], reverse=True))
    
    return sorted_skills, len(skills_found)

def process_resume(pdf_path: Path, category: str) -> Dict[str, Any]:
    """Process a single resume PDF and extract key features."""
    try:
        # Extract text from PDF
        processor = PDFProcessor()
        raw_text = processor.extract_text_from_pdf(str(pdf_path))
        
        if not raw_text or not raw_text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return None
        
        # Extract skills with frequency
        skills_dict, total_skills = extract_skills_with_frequency(raw_text, list(DEFAULT_SKILLS))
        
        # Extract education information
        education_info = extract_education_info(raw_text, EDUCATION_KEYWORDS)
        education_text = ' | '.join(education_info) if education_info else ''
        
        # Extract marks/CGPA from education section
        marks = extract_marks(education_text)
        
        # Get top 5 skills by frequency
        top_skills = list(skills_dict.keys())[:5]
        
        return {
            'filepath': str(pdf_path),
            'filename': pdf_path.name,
            'category': category,
            'text_length': len(raw_text),
            'total_skills': total_skills,
            'skills_found': ', '.join(skills_dict.keys()),
            'skills_frequency': json.dumps(skills_dict),
            'top_skills': ', '.join(top_skills),
            'education_info': education_text,
            'cgpa': marks.get('cgpa', None),
            'percentage': marks.get('percentage', None),
            'num_education': len(education_info),
            'has_education': 1 if education_info else 0,
            'has_marks': 1 if marks else 0
        }
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        if 'raw_text' in locals():
            print(f"Text length: {len(raw_text)} characters")
        return None

def process_resume_directory(data_dir: Path) -> pd.DataFrame:
    """Process all resumes in the given directory and its subdirectories."""
    data = []
    
    # Get all PDF files in the directory and its subdirectories
    pdf_files = list(data_dir.glob('**/*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    # Process each PDF file
    for pdf_path in tqdm(pdf_files, desc="Processing resumes"):
        # The category is the name of the parent directory
        category = pdf_path.parent.name.replace('-', ' ').title()
        
        # Process the resume
        result = process_resume(pdf_path, category)
        if result:
            data.append(result)
    
    # Create a DataFrame from the processed data
    if not data:
        print("No valid resumes were processed.")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    print(f"\nProcessed {len(df)} resumes from {len(pdf_files)} PDF files")
    print("Categories found:", df['category'].unique().tolist())
    
    return df

def save_to_csv(df: pd.DataFrame, output_file: str) -> None:
    """Save the processed data to a CSV file."""
    if df.empty:
        print("No data to save.")
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    print(f"Total resumes processed: {len(df)}")

def main():
    # Define paths
    data_dir = Path("../DATA/data")  # Update this to your data directory
    output_file = "../data/processed/resumes.csv"
    
    # Process all resumes
    df = process_resume_directory(data_dir)
    
    # Save to CSV
    if not df.empty:
        save_to_csv(df, output_file)

if __name__ == "__main__":
    main()
