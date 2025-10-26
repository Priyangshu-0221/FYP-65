"""Static internship catalog used for recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import json


@dataclass(slots=True)
class Internship:
    id: str
    title: str
    company: str
    location: str
    category: str
    skills: list[str]
    description: str
    apply_link: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "category": self.category,
            "skills": self.skills,
            "description": self.description,
            "apply_link": self.apply_link,
        }


def _load_default_catalog() -> list[Internship]:
    """Return the built-in internship catalog."""
    return [
        Internship(
            id="ds-ml-001",
            title="Data Science Intern",
            company="Insight Analytics",
            location="Remote",
            category="data science",
            skills=["python", "machine learning", "statistics", "sql"],
            description="Assist the analytics team with building predictive models and dashboards.",
            apply_link="https://example.com/internships/data-science",
        ),
        Internship(
            id="sw-fe-002",
            title="Frontend Engineering Intern",
            company="Pixel Labs",
            location="Bengaluru, IN",
            category="web development",
            skills=["javascript", "react", "html", "css"],
            description="Build user-facing features in React and collaborate with designers on UI/UX polishing.",
            apply_link="https://example.com/internships/frontend",
        ),
        Internship(
            id="sw-be-003",
            title="Backend Engineering Intern",
            company="CloudStack",
            location="Hyderabad, IN",
            category="software engineering",
            skills=["python", "sql", "aws", "docker"],
            description="Design REST APIs and support microservices deployments on AWS.",
            apply_link="https://example.com/internships/backend",
        ),
        Internship(
            id="an-bi-004",
            title="Business Intelligence Intern",
            company="MarketPulse",
            location="Remote",
            category="business analyst",
            skills=["excel", "sql", "data analysis", "power bi"],
            description="Support business analytics stakeholders with ad-hoc reporting and dashboard automation.",
            apply_link="https://example.com/internships/bi",
        ),
        Internship(
            id="cy-sec-005",
            title="Cybersecurity Intern",
            company="SecureNet",
            location="Gurgaon, IN",
            category="cyber security",
            skills=["networking", "python", "linux", "penetration testing"],
            description="Work with the security operations team on vulnerability assessments and monitoring.",
            apply_link="https://example.com/internships/cybersecurity",
        ),
        Internship(
            id="pm-prd-006",
            title="Product Management Intern",
            company="LaunchPad",
            location="Remote",
            category="product management",
            skills=["communication", "project management", "user research", "analytics"],
            description="Partner with cross-functional teams to shape product requirements and product metrics.",
            apply_link="https://example.com/internships/product",
        ),
        Internship(
            id="ux-des-007",
            title="UX Design Intern",
            company="DesignHive",
            location="Pune, IN",
            category="ui/ux",
            skills=["figma", "user research", "prototyping", "communication"],
            description="Collaborate with product designers to create wireframes and conduct usability testing.",
            apply_link="https://example.com/internships/ux",
        ),
        Internship(
            id="ai-re-008",
            title="AI Research Intern",
            company="DeepMindset",
            location="Remote",
            category="artificial intelligence",
            skills=["python", "deep learning", "research", "pytorch"],
            description="Explore novel deep learning architectures and contribute to open-source experiments.",
            apply_link="https://example.com/internships/ai-research",
        ),
    ]


def load_catalog(from_json: Path | None = None) -> list[Internship]:
    """Load internship catalog from a JSON file or fall back to defaults."""
    if from_json is None:
        return _load_default_catalog()

    if not from_json.exists():
        raise FileNotFoundError(f"Internship catalog file not found: {from_json}")

    with from_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    internships: list[Internship] = []
    for item in data:
        internships.append(
            Internship(
                id=item["id"],
                title=item["title"],
                company=item.get("company", ""),
                location=item.get("location", ""),
                category=item.get("category", ""),
                skills=list(item.get("skills", [])),
                description=item.get("description", ""),
                apply_link=item.get("apply_link", ""),
            )
        )
    return internships


def to_serializable(internships: Iterable[Internship]) -> List[dict]:
    """Convert Internship objects to dictionaries for JSON responses."""
    return [internship.to_dict() for internship in internships]
