"""Pipeline for extracting structured features from PDF resumes."""

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from PyPDF2 import PdfReader

# ---------------------------------------------------------------------------
# Reference data: tweak these lists to fit domain-specific requirements.
# ---------------------------------------------------------------------------
DEFAULT_SKILLS: Sequence[str] = (
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "sql",
    "excel",
    "power bi",
    "tableau",
    "machine learning",
    "deep learning",
    "data analysis",
    "project management",
    "communication",
    "javascript",
    "html",
    "css",
    "react",
    "node",
    "aws",
    "azure",
    "docker",
    "kubernetes",
)

EDUCATION_KEYWORDS: Sequence[str] = (
    "b.tech",
    "b.e",
    "bachelor",
    "master",
    "m.tech",
    "mba",
    "mca",
    "bsc",
    "msc",
    "phd",
    "diploma",
    "university",
    "college",
    "degree",
)

# Regexes capturing common marks and grade representations.
MARK_PATTERNS: Sequence[re.Pattern] = (
    re.compile(r"\b\d{2,3}\s?%"),
    re.compile(r"\b\d\.\d{1,2}\s*(?:gpa|cgpa)\b"),
    re.compile(r"\b(?:cgpa|gpa)\s*[:=-]?\s*\d\.\d{1,2}\b"),
)


# ---------------------------------------------------------------------------
# Core PDF and text utilities.
# ---------------------------------------------------------------------------

def list_pdf_files(pdf_dir: Path) -> List[Path]:
    """Return all PDF files in the given directory (non-recursive)."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory {pdf_dir} does not exist")
    return sorted(path for path in pdf_dir.iterdir() if path.suffix.lower() == ".pdf")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Read a PDF and return all text as a single string."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation/digits, and compress whitespace."""
    lowered = text.lower()
    letters_only = re.sub(r"[^a-z\s]", " ", lowered)
    normalized = re.sub(r"\s+", " ", letters_only)
    return normalized.strip()


def extract_skills(cleaned_text: str, skills: Iterable[str]) -> List[str]:
    """Identify skills present in the cleaned resume text."""
    found = set()
    for skill in skills:
        pattern = rf"\b{re.escape(skill.lower())}\b"
        if re.search(pattern, cleaned_text):
            found.add(skill)
    return sorted(found)


def extract_education(raw_text: str, keywords: Iterable[str]) -> str:
    """Return lines or snippets mentioning education keywords."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    hits = [line for line in lines if any(keyword in line.lower() for keyword in keywords)]
    return " | ".join(hits)


def extract_marks(raw_text: str) -> str:
    """Search for marks/grade mentions (percentage, GPA, CGPA)."""
    lowered = raw_text.lower()
    matches = set()
    for pattern in MARK_PATTERNS:
        matches.update(pattern.findall(lowered))
    # Normalize common GPA formats back to uppercase label for readability.
    formatted = [match.upper() if "gpa" in match else match for match in matches]
    return " | ".join(sorted(formatted))


def extract_features_from_pdf(
    pdf_path: Path,
    skills: Iterable[str],
    education_keywords: Iterable[str],
) -> dict:
    """Process a single PDF and return extracted fields."""
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)
    skills_found = extract_skills(cleaned, skills)
    education_info = extract_education(raw_text, education_keywords)
    marks_info = extract_marks(raw_text)
    return {
        "Filename": pdf_path.name,
        "Skills": ", ".join(skills_found),
        "Skills_Count": len(skills_found),
        "Education": education_info,
        "Marks": marks_info,
    }


def process_pdf_directory(pdf_dir: Path) -> pd.DataFrame:
    """Process every PDF resume in the directory and return a DataFrame."""
    pdf_files = list_pdf_files(pdf_dir)
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    records = [
        extract_features_from_pdf(
            pdf_path,
            skills=DEFAULT_SKILLS,
            education_keywords=EDUCATION_KEYWORDS,
        )
        for pdf_path in pdf_files
    ]
    return pd.DataFrame.from_records(records)


def save_outputs(df: pd.DataFrame, output_csv: Path, output_excel: Path) -> None:
    """Persist the structured data to CSV and Excel files."""
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)


def display_sample(df: pd.DataFrame, rows: int = 5) -> None:
    """Show a sample of the DataFrame for quick verification."""
    print("\nSample of extracted resume features:\n")
    print(df.head(rows).to_string(index=False))


# ---------------------------------------------------------------------------
# Command-line interface.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features from PDF resumes")
    parser.add_argument("--pdf-dir", type=Path, required=True, help="Directory containing resume PDFs")
    parser.add_argument("--output-csv", type=Path, default=Path("pdf_resumes_features.csv"), help="CSV output path")
    parser.add_argument("--output-excel", type=Path, default=Path("pdf_resumes_features.xlsx"), help="Excel output path")
    parser.add_argument("--sample-rows", type=int, default=5, help="Number of rows to display after processing")
    return parser.parse_args()


def main() -> None:
    """Entry point to run the PDF feature extraction pipeline."""
    args = parse_args()

    features_df = process_pdf_directory(args.pdf_dir)
    save_outputs(features_df, args.output_csv, args.output_excel)
    display_sample(features_df, rows=args.sample_rows)


if __name__ == "__main__":
    main()
