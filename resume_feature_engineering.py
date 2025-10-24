"""Utility module for enriching resume datasets with structured features."""

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas as pd

# Predefined reference data ----------------------------------------------------
DEFAULT_SKILLS: Set[str] = {
    "python",
    "java",
    "c++",
    "sql",
    "excel",
    "machine learning",
    "deep learning",
    "project management",
    "communication",
    "data analysis",
    "statistics",
    "javascript",
    "html",
    "css",
    "react",
    "node",
    "aws",
    "azure",
    "docker",
    "kubernetes",
}

EDUCATION_KEYWORDS: Sequence[str] = (
    "bachelor",
    "master",
    "phd",
    "b.tech",
    "b.e",
    "m.tech",
    "mba",
    "mca",
    "bsc",
    "msc",
    "diploma",
    "university",
    "college",
    "degree",
)

EXPERIENCE_KEYWORDS: Sequence[str] = (
    "experience",
    "worked",
    "responsible",
    "managed",
    "led",
    "developed",
    "years",
    "projects",
    "team",
    "implemented",
)


# Core processing helpers ------------------------------------------------------

def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the resume CSV file into a DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file {csv_path} does not exist")
    df = pd.read_csv(csv_path)
    missing_cols = {"Category", "Resume"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV must contain columns: {missing_cols}")
    return df.fillna({"Resume": ""})


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation/digits, and squeeze whitespace."""
    lowered = text.lower()
    letters_only = re.sub(r"[^a-z\s]", " ", lowered)
    normalized = re.sub(r"\s+", " ", letters_only)
    return normalized.strip()


def tokenize_sentences(text: str) -> List[str]:
    """Split raw text into simple sentence-like chunks using punctuation."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_skills(text: str, skills: Iterable[str]) -> List[str]:
    """Find unique skills present in the resume text based on the reference list."""
    normalized_text = f" {text.lower()} "
    found = set()
    for skill in skills:
        pattern = rf"\b{re.escape(skill.lower())}\b"
        if re.search(pattern, normalized_text):
            found.add(skill)
    return sorted(found)


def extract_education_info(text: str, keywords: Iterable[str]) -> str:
    """Return lines mentioning education-related keywords."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    education_lines = [line for line in lines if any(keyword in line.lower() for keyword in keywords)]
    return " | ".join(education_lines)


def extract_experience_sentences(text: str, keywords: Iterable[str]) -> str:
    """Return sentences that contain experience-related keywords."""
    sentences = tokenize_sentences(text)
    experience_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return " | ".join(experience_sentences)


def augment_dataset(
    df: pd.DataFrame,
    skills: Iterable[str],
    education_keywords: Iterable[str],
    experience_keywords: Iterable[str],
) -> pd.DataFrame:
    """Add engineered features to the resume DataFrame."""
    df = df.copy()
    df["Cleaned_Resume"] = df["Resume"].apply(clean_text)
    df["Skills_Found"] = df["Cleaned_Resume"].apply(lambda text: extract_skills(text, skills))
    df["Skills_Count"] = df["Skills_Found"].apply(len)
    df["Education_Info"] = df["Resume"].apply(lambda text: extract_education_info(text, education_keywords))
    df["Experience_Sentences"] = df["Resume"].apply(lambda text: extract_experience_sentences(text, experience_keywords))
    return df


def save_outputs(df: pd.DataFrame, output_csv: Path, output_excel: Path) -> None:
    """Persist the enriched DataFrame to CSV and Excel formats."""
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)


# Optional utilities -----------------------------------------------------------

def load_skill_list(skill_file: Path | None) -> Set[str]:
    """Load user-provided skills from a newline-separated text file."""
    if skill_file is None:
        return set(DEFAULT_SKILLS)
    if not skill_file.exists():
        raise FileNotFoundError(f"Skill file {skill_file} does not exist")
    skills = {line.strip().lower() for line in skill_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    return skills


def display_sample(df: pd.DataFrame, rows: int = 5) -> None:
    """Print a small sample of the enriched DataFrame."""
    sample = df.head(rows)
    print("\nSample of enriched dataset:\n")
    print(sample.to_string(index=False))


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
