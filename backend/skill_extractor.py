"""Skill extraction utilities leveraging existing feature engineering logic."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from resume_feature_engineering import (
    DEFAULT_SKILLS,
    EDUCATION_KEYWORDS,
    EXPERIENCE_KEYWORDS,
    augment_dataset,
)
from resume_classifier import ensure_nltk_dependencies, preprocess_corpus


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
