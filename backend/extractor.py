"""
Simple PDF Text Extractor
Extracts text, skills, education, experience, and academic scores from resumes
"""

import re
from pathlib import Path
from typing import Dict, List, Set
from PyPDF2 import PdfReader

# Comprehensive skills list
SKILLS = {
    'python', 'java', 'c++', 'javascript', 'typescript', 'react', 'angular', 'vue',
    'node', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring', 'html', 'css',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes', 'aws',
    'azure', 'gcp', 'git', 'github', 'linux', 'windows', 'machine learning', 'ml',
    'deep learning', 'data science', 'artificial intelligence', 'ai', 'nlp',
    'computer vision', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
    'matplotlib', 'seaborn', 'excel', 'powerpoint', 'word', 'communication',
    'leadership', 'teamwork', 'problem solving', 'analytical', 'accounting',
    'finance', 'tally', 'quickbooks', 'sap', 'oracle', 'taxation', 'audit',
    'financial analysis', 'budgeting', 'forecasting', 'excel vba', 'power bi',
    'tableau', 'data analysis', 'statistics', 'r programming', 'matlab'
}

# Education keywords
EDUCATION_KEYWORDS = [
    'bachelor', 'master', 'phd', 'b.tech', 'b.e', 'm.tech', 'mba', 'mca',
    'bsc', 'msc', 'bba', 'bcom', 'mcom', 'diploma', 'university', 'college',
    'degree', 'education', 'gpa', 'cgpa', 'percentage', 'grade'
]

# Experience keywords
EXPERIENCE_KEYWORDS = [
    'experience', 'work', 'internship', 'job', 'position', 'role',
    'worked', 'responsible', 'managed', 'led', 'developed', 'years',
    'company', 'organization', 'project', 'team'
]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""


def extract_skills(text: str) -> List[str]:
    """Extract skills from text"""
    text_lower = text.lower()
    found_skills = []
    
    for skill in SKILLS:
        # Use word boundary to match whole words
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    return sorted(list(set(found_skills)))


def extract_education(text: str) -> str:
    """Extract education information"""
    lines = text.split('\n')
    education_lines = []
    
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in EDUCATION_KEYWORDS):
            education_lines.append(line.strip())
    
    return ' | '.join(education_lines[:5])  # Return top 5 education lines


def extract_experience(text: str) -> str:
    """Extract experience information"""
    lines = text.split('\n')
    experience_lines = []
    
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in EXPERIENCE_KEYWORDS):
            experience_lines.append(line.strip())
    
    return ' | '.join(experience_lines[:5])  # Return top 5 experience lines


def extract_academic_scores(text: str) -> Dict[str, str]:
    """Extract CGPA and percentage from text"""
    scores = {}
    
    # CGPA patterns
    cgpa_patterns = [
        r'cgpa[:\s]*([0-9]+\.?[0-9]*)',
        r'gpa[:\s]*([0-9]+\.?[0-9]*)',
        r'([0-9]+\.?[0-9]*)\s*/\s*10',
        r'([0-9]+\.?[0-9]*)\s*/\s*4'
    ]
    
    for pattern in cgpa_patterns:
        match = re.search(pattern, text.lower())
        if match:
            scores['cgpa'] = match.group(1)
            break
    
    # Percentage patterns
    percent_patterns = [
        r'percentage[:\s]*([0-9]+\.?[0-9]*)',
        r'([0-9]{2,3}\.?[0-9]*)\s*%',
        r'([0-9]{2,3}\.?[0-9]*)\s*percent'
    ]
    
    for pattern in percent_patterns:
        match = re.search(pattern, text.lower())
        if match:
            scores['percentage'] = match.group(1)
            break
    
    return scores


def process_resume(pdf_path: str, category: str = "") -> Dict:
    """Process a single resume and extract all information"""
    print(f"Processing: {Path(pdf_path).name}")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        return None
    
    # Extract all information
    skills = extract_skills(text)
    education = extract_education(text)
    experience = extract_experience(text)
    scores = extract_academic_scores(text)
    
    return {
        'filename': Path(pdf_path).name,
        'category': category,
        'text': text,
        'skills': skills,
        'skills_count': len(skills),
        'education': education,
        'experience': experience,
        'cgpa': scores.get('cgpa', ''),
        'percentage': scores.get('percentage', '')
    }


if __name__ == "__main__":
    # Test with a single PDF
    test_pdf = r"c:\Users\abhij\Desktop\Test\FYP-65\DATA\data\ACCOUNTANT\10554236.pdf"
    
    if Path(test_pdf).exists():
        result = process_resume(test_pdf, "ACCOUNTANT")
        if result:
            print("\n=== Extraction Results ===")
            print(f"Skills: {result['skills'][:10]}")  # First 10 skills
            print(f"Skills Count: {result['skills_count']}")
            print(f"CGPA: {result['cgpa']}")
            print(f"Percentage: {result['percentage']}")
            print(f"Education: {result['education'][:100]}...")
            print(f"Experience: {result['experience'][:100]}...")
    else:
        print("Test PDF not found")
