"""Skill extraction utilities leveraging existing feature engineering logic."""

from __future__ import annotations

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathlib import Path
from typing import Iterable, List, Set, Sequence, Union, Optional

import pandas as pd

def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation/digits, and squeeze whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove non-letters
    text = re.sub(r'\s+', ' ', text)        # Normalize whitespace
    return text.strip()

def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    processed_tokens = []
    
    for token in tokens:
        if token not in stop_words and token.isalnum():
            processed_tokens.append(lemmatizer.lemmatize(token))
    
    return ' '.join(processed_tokens)

def ensure_nltk_dependencies():
    """Ensure all required NLTK resources are downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

def preprocess_corpus(text_series: pd.Series) -> pd.Series:
    """Preprocess a series of text documents."""
    ensure_nltk_dependencies()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return text_series.apply(lambda x: ' '.join(
        lemmatizer.lemmatize(word) for word in word_tokenize(x.lower())
        if word.isalnum() and word not in stop_words
    ))

# Define these constants locally to avoid circular imports
DEFAULT_SKILLS: Set[str] = {
    'python', 'java', 'c++', 'javascript', 'sql', 'html', 'css',
    'machine learning', 'data analysis', 'project management',
    'team leadership', 'problem solving', 'communication', 'cloud computing',
    'aws', 'azure', 'docker', 'kubernetes', 'react', 'node', 'statistics',
    'excel', 'deep learning'
}

EDUCATION_KEYWORDS: Sequence[str] = (
    'bachelor', 'master', 'phd', 'b.tech', 'b.e', 'm.tech', 'mba', 'mca',
    'bsc', 'msc', 'diploma', 'university', 'college', 'degree', 'education',
    'coursework', 'gpa', 'graduate', 'undergraduate'
)

EXPERIENCE_KEYWORDS: Sequence[str] = (
    'experience', 'work', 'internship', 'job', 'position', 'role',
    'responsibilities', 'achievements', 'projects', 'employment',
    'worked', 'responsible', 'managed', 'led', 'developed', 'years', 'team',
    'implemented'
)

def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation/digits, and squeeze whitespace."""
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    letters_only = re.sub(r'[^a-z\s]', ' ', lowered)
    normalized = re.sub(r'\s+', ' ', letters_only)
    return normalized.strip()

def tokenize_sentences(text: str) -> List[str]:
    """Split raw text into simple sentence-like chunks using punctuation."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def extract_skills(text: str, skills: Iterable[str]) -> List[str]:
    """Find unique skills present in the resume text based on the reference list."""
    if not isinstance(text, str):
        return []
    normalized_text = f" {text.lower()} "
    found = set()
    for skill in skills:
        pattern = rf'\b{re.escape(skill.lower())}\b'
        if re.search(pattern, normalized_text):
            found.add(skill)
    return sorted(found)

def extract_education_info(text: str, keywords: Iterable[str]) -> str:
    """Return lines mentioning education-related keywords."""
    if not isinstance(text, str):
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    education_lines = [line for line in lines if any(keyword in line.lower() for keyword in keywords)]
    return " | ".join(education_lines)

def extract_experience_sentences(text: str, keywords: Iterable[str]) -> str:
    """Return sentences that contain experience-related keywords."""
    if not isinstance(text, str):
        return ""
    sentences = tokenize_sentences(text)
    experience_sentences = [sentence for sentence in sentences 
                          if any(keyword in sentence.lower() for keyword in keywords)]
    return " | ".join(experience_sentences)

def augment_dataset(
    df: pd.DataFrame,
    skills: Iterable[str] = None,
    education_keywords: Iterable[str] = None,
    experience_keywords: Iterable[str] = None,
) -> pd.DataFrame:
    """Add engineered features to the resume DataFrame.
    
    Args:
        df: Input DataFrame with at least a 'Resume' column
        skills: Collection of skills to search for in resumes
        education_keywords: Keywords to identify education-related content
        experience_keywords: Keywords to identify experience-related content
        
    Returns:
        DataFrame with additional feature columns
    """
    if df.empty or 'Resume' not in df.columns:
        return df
        
    df = df.copy()
    skills = set(skills or DEFAULT_SKILLS)
    edu_keywords = set(education_keywords or EDUCATION_KEYWORDS)
    exp_keywords = set(experience_keywords or EXPERIENCE_KEYWORDS)
    
    # Clean and process the resume text
    df['Cleaned_Resume'] = df['Resume'].fillna('').astype(str).apply(clean_text)
    
    # Extract skills and other features
    df['Skills_Found'] = df['Cleaned_Resume'].apply(lambda x: extract_skills(x, skills))
    df['Skills_Count'] = df['Skills_Found'].apply(len)
    df['Education_Info'] = df['Resume'].fillna('').astype(str).apply(
        lambda x: extract_education_info(x, edu_keywords)
    )
    df['Experience_Sentences'] = df['Resume'].fillna('').astype(str).apply(
        lambda x: extract_experience_sentences(x, exp_keywords)
    )
    
    return df


def extract_skills_from_text(resume_text: str, skills_vocab: Iterable[str] | None = None) -> list[str]:
    """Return a sorted list of skills detected within the resume text."""
    skills = set(skills_vocab or DEFAULT_SKILLS)
    ensure_nltk_dependencies()
    processed_text = preprocess_corpus(pd.Series([resume_text]))[0]
    enriched_df = augment_dataset(
        pd.DataFrame({"Category": ["unknown"], "Resume": [resume_text]}),
        skills=skills,
        education_keywords=EDUCATION_KEYWORDS,
        experience_keywords=EXPERIENCE_KEYWORDS,
    )
    return enriched_df.loc[0, "Skills_Found"]


def extract_skills_from_dataframe(df: pd.DataFrame, skills_vocab: Iterable[str] | None = None) -> list[list[str]]:
    """Extract skills for each resume row in an existing dataframe."""
    skills = set(skills_vocab or DEFAULT_SKILLS)
    ensure_nltk_dependencies()
    enriched_df = augment_dataset(
        df,
        skills=skills,
        education_keywords=EDUCATION_KEYWORDS,
        experience_keywords=EXPERIENCE_KEYWORDS,
    )
    return enriched_df["Skills_Found"].tolist()
