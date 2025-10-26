"""Recommendation engine for matching resumes to internships."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .internship_catalog import Internship


def score_internship(skills: Sequence[str], internship: Internship) -> float:
    """Compute a simple relevance score based on overlapping skills."""
    if not skills:
        return 0.0
    skill_set = set(skill.lower() for skill in skills)
    internship_skills = set(skill.lower() for skill in internship.skills)
    overlap = skill_set & internship_skills
    if not overlap:
        return 0.0
    # Jaccard similarity
    union_count = len(skill_set | internship_skills)
    score = len(overlap) / union_count
    return float(score)


def rank_internships(skills: Sequence[str], internships: Iterable[Internship], top_k: int = 5) -> list[tuple[Internship, float]]:
    """Rank internships by relevance to the provided skill list."""
    scored = []
    for internship in internships:
        score = score_internship(skills, internship)
        if score > 0:
            scored.append((internship, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
