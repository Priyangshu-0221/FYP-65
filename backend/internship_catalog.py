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
    skills: list
    description: str
    apply_link: str = "#"
    domain: str = ""
    skills_required: list = None
    academic_requirements: dict = None

    def __post_init__(self):
        # For backward compatibility
        if self.skills_required is None:
            self.skills_required = [{"name": skill, "proficiency": "intermediate"} for skill in self.skills]
        if not self.domain and self.category:
            self.domain = self.category
        if self.academic_requirements is None:
            self.academic_requirements = {
                "min_cgpa": 3.0,
                "min_percentage": 65.0,
                "preferred_majors": ["Computer Science", "Information Technology", "Related field"]
            }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "category": self.category,
            "domain": self.domain,
            "skills": [s['name'] if isinstance(s, dict) else s for s in self.skills_required],
            "skills_required": self.skills_required,
            "description": self.description,
            "apply_link": self.apply_link,
            "academic_requirements": self.academic_requirements
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
    if from_json and from_json.exists():
        try:
            with open(from_json, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "internships" in data:
                    data = data["internships"]
                
                internships = []
                for item in data:
                    try:
                        # Handle both old and new format
                        skills = item.get("skills_required", item.get("skills", []))
                        
                        internship = Internship(
                            id=item.get("id", str(hash(json.dumps(item, sort_keys=True)))[:8]),
                            title=item.get("title", ""),
                            company=item.get("company", ""),
                            location=item.get("location", ""),
                            category=item.get("category", item.get("domain", "")),
                            domain=item.get("domain", item.get("category", "")),
                            skills=skills,  # This will be processed in __post_init__
                            description=item.get("description", ""),
                            apply_link=item.get("apply_link", "#"),
                            skills_required=skills,
                            academic_requirements=item.get("academic_requirements", {
                                "min_cgpa": item.get("min_cgpa", 3.0),
                                "min_percentage": item.get("min_percentage", 65.0),
                                "preferred_majors": item.get("preferred_majors", ["Computer Science", "Related field"])
                            })
                        )
                        internships.append(internship)
                    except Exception as e:
                        print(f"Error processing internship: {e}")
                        continue
                
                return internships
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading catalog from {from_json}: {e}")
            print("Falling back to default catalog.")

    # Fall back to default catalog if loading from JSON fails or no path provided
    return _load_default_catalog()


def to_serializable(internships: Iterable[Internship]) -> List[dict]:
    """Convert Internship objects to dictionaries for JSON responses."""
    return [internship.to_dict() for internship in internships]
