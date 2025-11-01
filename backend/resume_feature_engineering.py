"""
Resume Feature Engineering Module

This module provides comprehensive functionality for processing and enriching resume data,
supporting both text-based and PDF-based resumes. It includes utilities for:
- Text extraction from PDFs
- Skill extraction and normalization
- Education and experience detection
- Feature engineering for machine learning
- Data cleaning and preprocessing

Key Features:
- Support for multiple input formats (text, PDF)
- Configurable skill and keyword dictionaries
- Extensible architecture for custom processing
- Efficient batch processing of multiple resumes

Example Usage:
    >>> from pathlib import Path
    >>> from resume_feature_engineering import process_resumes
    >>> 
    >>> # Process a directory of PDF resumes
    >>> df = process_resumes(Path("path/to/resumes"))
    >>> 
    >>> # Process a single text resume
    >>> features = extract_features_from_text("Python developer with 5 years of experience...")
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader

# Predefined reference data ----------------------------------------------------
DEFAULT_SKILLS: Set[str] = {
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    
    # Web Technologies
    "html", "css", "sass", "less", "react", "angular", "vue", "django",
    "flask", "fastapi", "node.js", "express", "spring", "asp.net", "laravel",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github actions", "gitlab ci", "circleci", "argocd",
    
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "cassandra", "dynamodb",
    "oracle", "sql server", "sqlite", "firebase", "cosmosdb",
    
    # Data Science & ML
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "pyspark", "hadoop", "hive", "kafka", "airflow", "mlflow", "kubeflow",
    
    # Tools & Platforms
    "git", "linux", "bash", "powershell", "jira", "confluence", "slack",
    "docker", "kubernetes", "terraform", "ansible", "puppet", "chef",
    
    # Soft Skills
    "problem solving", "teamwork", "leadership", "communication",
    "project management", "agile", "scrum", "kanban", "devops",
    "continuous integration", "continuous deployment"
}

EDUCATION_KEYWORDS: Sequence[str] = (
    # Degree Types
    "bachelor", "master", "phd", "doctorate", "associate", "diploma",
    "b.tech", "b.e.", "b.eng", "b.sc.", "bca", "bca", "b.com", "b.a.",
    "m.tech", "m.e.", "m.eng", "m.sc.", "mca", "mba", "msc", "ms", "ma",
    "ph.d", "d.phil", "postdoc", "post-doc",
    
    # Education Institutions
    "university", "college", "institute", "school", "academy", "faculty",
    "higher education", "undergraduate", "postgraduate", "graduate",
    
    # Education Related
    "degree", "major", "minor", "thesis", "dissertation", "gpa", "cgpa",
    "grade", "honors", "cum laude", "magna cum laude", "summa cum laude",
    "dean's list", "scholarship", "fellowship", "research", "publications"
)

EXPERIENCE_KEYWORDS: Sequence[str] = (
    "experience", "worked", "employed", "internship", "intern", "freelance",
    "contract", "full-time", "part-time", "remote", "on-site", "hybrid",
    "years", "yrs", "months", "mos", "responsibilities", "achievements",
    "projects", "technologies", "tools", "frameworks", "libraries",
    "led", "managed", "developed", "implemented", "designed", "created",
    "optimized", "improved", "reduced", "increased", "achieved", "delivered"
)

# =============================================================================
# Core Text Processing Functions
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text by:
    1. Converting to lowercase
    2. Removing special characters and extra whitespace
    3. Normalizing unicode characters
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Replace email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Replace phone numbers
    text = re.sub(r'\b(?:\+?[\d\s-]+\([\d\s-]+\)?[\d\s-]+|\d{10,})\b', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\[\]{}]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text into words with optional stopword removal.
    
    Args:
        text: Input text to tokenize
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of tokens
    """
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
    
    return tokens


def extract_skills(text: str, skills: Iterable[str] = None) -> Set[str]:
    """
    Extract skills from text based on a predefined list of skills.
    
    Args:
        text: Input text to extract skills from
        skills: Optional custom list of skills to match against
        
    Returns:
        Set of matched skills
    """
    if skills is None:
        skills = DEFAULT_SKILLS
    
    # Clean and tokenize text
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text, remove_stopwords=True)
    
    # Convert to set for faster lookups
    skill_set = set(s.lower() for s in skills)
    
    # Find matches (including n-grams up to 3 words)
    found_skills = set()
    
    # Check unigrams
    found_skills.update(skill for skill in skill_set if skill in cleaned_text)
    
    # Check bigrams and trigrams
    for n in [2, 3]:
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        found_skills.update(skill for skill in skill_set if skill in ngrams)
    
    return found_skills


def extract_education(text: str, keywords: Iterable[str] = None) -> List[str]:
    """
    Extract education-related information from text.
    
    Args:
        text: Input text to extract education info from
        keywords: Optional custom list of education-related keywords
        
    Returns:
        List of education-related sentences
    """
    if keywords is None:
        keywords = EDUCATION_KEYWORDS
    
    # Clean the text
    text = clean_text(text)
    
    # Split into sentences
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    sentences = nltk.sent_tokenize(text)
    
    # Find education-related sentences
    education_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            education_sentences.append(sentence.strip())
    
    return education_sentences


def extract_experience(text: str, keywords: Iterable[str] = None) -> Dict[str, any]:
    """
    Extract work experience information from text.
    
    Args:
        text: Input text to extract experience from
        keywords: Optional custom list of experience-related keywords
        
    Returns:
        Dictionary containing experience information
    """
    if keywords is None:
        keywords = EXPERIENCE_KEYWORDS
    
    # Clean the text
    text = clean_text(text)
    
    # Extract years of experience
    experience_years = 0
    year_matches = re.findall(r'(\d+)\s*(?:years?|yrs?|\+)', text)
    if year_matches:
        experience_years = max(int(y) for y in year_matches)
    
    # Extract job titles (simplified)
    job_titles = []
    title_patterns = [
        r'(?:worked|served|acted)\s+as\s+(?:a\s+)?([\w\s]+?(?=\s+at\s|\s+for\s|\s+in\s|,|;|\.|$))',
        r'(?:position|role|title)[\s:]+([\w\s]+?(?=\s+at\s|\s+for\s|\s+in\s|,|;|\.|$))'
    ]
    
    for pattern in title_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        job_titles.extend(m.strip() for m in matches if len(m.strip().split()) <= 4)
    
    # Extract companies (simplified)
    companies = []
    company_matches = re.findall(
        r'(?:at|in|from|with|,|;|\bat\b)\s*([A-Z][\w\s&,.()-]+?(?=\s+(?:from|to|\d|$))',
        text
    )
    companies = [c.strip() for c in company_matches if len(c.strip().split()) <= 5]
    
    return {
        'years_experience': experience_years,
        'job_titles': list(set(job_titles)),
        'companies': list(set(companies)),
        'experience_sentences': [s for s in text.split('.') if any(kw in s.lower() for kw in keywords)]
    }


# =============================================================================
# PDF Processing Functions
# =============================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""


def process_pdf_resume(pdf_path: Path) -> Dict[str, any]:
    """
    Process a single PDF resume and extract structured information.
    
    Args:
        pdf_path: Path to the PDF resume
        
    Returns:
        Dictionary containing extracted resume information
    """
    # Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return {}
    
    # Extract features
    skills = extract_skills(raw_text)
    education = extract_education(raw_text)
    experience = extract_experience(raw_text)
    
    return {
        'filename': pdf_path.name,
        'text': raw_text,
        'skills': list(skills),
        'education': education,
        'experience': experience,
        'experience_years': experience.get('years_experience', 0),
        'job_titles': experience.get('job_titles', []),
        'companies': experience.get('companies', [])
    }


def process_resumes(directory: Path) -> pd.DataFrame:
    """
    Process all PDF resumes in a directory.
    
    Args:
        directory: Directory containing PDF resumes
        
    Returns:
        DataFrame with extracted resume information
    """
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory")
    
    # Find all PDF files
    pdf_files = list(directory.glob('*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return pd.DataFrame()
    
    # Process each PDF
    results = []
    for pdf_file in pdf_files:
        try:
            result = process_pdf_resume(pdf_file)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


# =============================================================================
# Main Functionality
# =============================================================================

def main():
    """Command-line interface for processing resumes."""
    parser = argparse.ArgumentParser(description='Process resumes and extract features')
    parser.add_argument('input_dir', type=Path, help='Directory containing PDF resumes')
    parser.add_argument('--output', type=Path, default='resume_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--skills', type=Path, help='Custom skills file (one per line)')
    
    args = parser.parse_args()
    
    # Load custom skills if provided
    custom_skills = None
    if args.skills and args.skills.exists():
        with open(args.skills, 'r', encoding='utf-8') as f:
            custom_skills = {line.strip().lower() for line in f if line.strip()}
    
    # Process resumes
    print(f"Processing resumes in {args.input_dir}...")
    df = process_resumes(args.input_dir)
    
    if df.empty:
        print("No valid resumes found or processed.")
        return
    
    # Save results
    output_path = args.output
    df.to_csv(output_path, index=False)
    print(f"Processed {len(df)} resumes. Results saved to {output_path}")


# =============================================================================
# Data Loading and Saving
# =============================================================================

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load resume data from a CSV file into a pandas DataFrame.
    
    Args:
        csv_path: Path to the input CSV file
        
    Returns:
        DataFrame containing the loaded resume data
        
    Raises:
        FileNotFoundError: If the input file does not exist
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file {csv_path} does not exist")
        
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required_columns = {"Category", "Resume"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV must contain columns: {missing_cols}")
    
    return df.fillna({"Resume": ""})


def augment_dataset(
    df: pd.DataFrame,
    skills: Iterable[str] = None,
    education_keywords: Iterable[str] = None,
    experience_keywords: Iterable[str] = None
) -> pd.DataFrame:
    """
    Add engineered features to the resume DataFrame.
    
    Args:
        df: Input DataFrame containing resume data
        skills: Optional list of skills to match against
        education_keywords: Optional list of education-related keywords
        experience_keywords: Optional list of experience-related keywords
        
    Returns:
        DataFrame with additional feature columns
    """
    if skills is None:
        skills = DEFAULT_SKILLS
    if education_keywords is None:
        education_keywords = EDUCATION_KEYWORDS
    if experience_keywords is None:
        experience_keywords = EXPERIENCE_KEYWORDS
    
    df = df.copy()
    
    # Extract features
    df['cleaned_text'] = df['Resume'].apply(clean_text)
    df['skills'] = df['Resume'].apply(lambda x: list(extract_skills(x, skills)))
    df['education'] = df['Resume'].apply(
        lambda x: extract_education(x, education_keywords)
    )
    df['experience'] = df['Resume'].apply(
        lambda x: extract_experience(x, experience_keywords)
    )
    
    return df


def save_outputs(df: pd.DataFrame, output_csv: Path, output_excel: Path) -> None:
    """
    Save the processed DataFrame to both CSV and Excel formats.
    
    Args:
        df: DataFrame to save
        output_csv: Path for CSV output
        output_excel: Path for Excel output
    """
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV and Excel
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)


# Command-line interface -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich resume dataset with structured features")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to input resume CSV")
    parser.add_argument("--output-csv", type=Path, default=Path("processed_resumes.csv"), help="Destination CSV path")
    parser.add_argument("--output-excel", type=Path, default=Path("processed_resumes.xlsx"), help="Destination Excel path")
    parser.add_argument("--skill-file", type=Path, default=None, help="Optional newline-separated skill list")
    parser.add_argument("--sample-rows", type=int, default=5, help="Number of rows to preview after processing")
    return parser.parse_args()


def main() -> None:
    """Entry point for command-line execution."""
    args = parse_args()

    df = load_dataset(args.input_csv)
    skills = load_skill_list(args.skill_file)

    enriched_df = augment_dataset(
        df,
        skills=skills,
        education_keywords=EDUCATION_KEYWORDS,
        experience_keywords=EXPERIENCE_KEYWORDS,
    )

    save_outputs(enriched_df, args.output_csv, args.output_excel)
    display_sample(enriched_df, rows=args.sample_rows)


if __name__ == "__main__":
    main()
