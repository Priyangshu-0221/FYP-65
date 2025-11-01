"""
Enhanced Dummy Recommendation Dataset Generator

This script generates a comprehensive dataset for testing internship recommendations.
The dataset includes varied internship positions with different skill requirements,
academic criteria, and other relevant attributes to test various CV matching scenarios.
"""

import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime, timedelta
import uuid

# Define the output path for the dataset
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "dummy_internship_recommendations.json"

# Comprehensive skills database with categories and proficiency levels
SKILLS_DATABASE = {
    "Programming Languages": {
        "Python": {"level": "Advanced", "category": "Core"},
        "JavaScript": {"level": "Intermediate", "category": "Core"},
        "Java": {"level": "Intermediate", "category": "Core"},
        "C++": {"level": "Advanced", "category": "Core"},
        "TypeScript": {"level": "Intermediate", "category": "Web"},
        "Go": {"level": "Beginner", "category": "Backend"},
        "Rust": {"level": "Beginner", "category": "Systems"},
        "Kotlin": {"level": "Intermediate", "category": "Mobile"},
        "Swift": {"level": "Intermediate", "category": "Mobile"},
        "SQL": {"level": "Intermediate", "category": "Database"}
    },
    "Frameworks & Libraries": {
        "React": {"level": "Intermediate", "category": "Frontend"},
        "Node.js": {"level": "Intermediate", "category": "Backend"},
        "Django": {"level": "Intermediate", "category": "Backend"},
        "Flask": {"level": "Intermediate", "category": "Backend"},
        "TensorFlow": {"level": "Advanced", "category": "ML"},
        "PyTorch": {"level": "Advanced", "category": "ML"},
        "Spring Boot": {"level": "Intermediate", "category": "Backend"},
        "React Native": {"level": "Intermediate", "category": "Mobile"},
        "Flutter": {"level": "Beginner", "category": "Mobile"},
        "Angular": {"level": "Intermediate", "category": "Frontend"}
    },
    "Tools & Platforms": {
        "Docker": {"level": "Intermediate", "category": "DevOps"},
        "Kubernetes": {"level": "Intermediate", "category": "DevOps"},
        "AWS": {"level": "Intermediate", "category": "Cloud"},
        "Azure": {"level": "Beginner", "category": "Cloud"},
        "GCP": {"level": "Beginner", "category": "Cloud"},
        "Git": {"level": "Intermediate", "category": "Version Control"},
        "Jenkins": {"level": "Beginner", "category": "CI/CD"},
        "GitHub Actions": {"level": "Beginner", "category": "CI/CD"},
        "Jira": {"level": "Beginner", "category": "Project Management"},
        "Tableau": {"level": "Intermediate", "category": "Data Visualization"}
    },
    "Data Science & ML": {
        "Pandas": {"level": "Intermediate", "category": "Data Analysis"},
        "NumPy": {"level": "Intermediate", "category": "Data Analysis"},
        "Scikit-learn": {"level": "Intermediate", "category": "ML"},
        "NLTK": {"level": "Beginner", "category": "NLP"},
        "spaCy": {"level": "Beginner", "category": "NLP"},
        "OpenCV": {"level": "Intermediate", "category": "Computer Vision"},
        "PySpark": {"level": "Beginner", "category": "Big Data"},
        "Hadoop": {"level": "Beginner", "category": "Big Data"},
        "Matplotlib": {"level": "Intermediate", "category": "Data Visualization"},
        "Seaborn": {"level": "Intermediate", "category": "Data Visualization"}
    },
    "Cybersecurity": {
        "Ethical Hacking": {"level": "Advanced", "category": "Offensive Security"},
        "Penetration Testing": {"level": "Advanced", "category": "Offensive Security"},
        "Network Security": {"level": "Intermediate", "category": "Defensive Security"},
        "Cryptography": {"level": "Advanced", "category": "Security"},
        "SIEM": {"level": "Intermediate", "category": "Security Operations"},
        "Incident Response": {"level": "Intermediate", "category": "Security Operations"},
        "OWASP Top 10": {"level": "Intermediate", "category": "Web Security"},
        "Kali Linux": {"level": "Intermediate", "category": "Tools"},
        "Wireshark": {"level": "Beginner", "category": "Networking"},
        "Metasploit": {"level": "Advanced", "category": "Penetration Testing"}
    }
}

# Domains with detailed descriptions and requirements
DOMAINS = {
    "Data Science": {
        "description": "Work with large datasets to extract insights and build predictive models",
        "required_skills": ["Python", "Pandas", "Machine Learning", "Statistics"],
        "academic_requirements": {"min_cgpa": 3.0, "min_percentage": 75, "preferred_majors": ["Computer Science", "Statistics", "Mathematics"]}
    },
    "Web Development": {
        "description": "Build responsive and scalable web applications",
        "required_skills": ["JavaScript", "React", "Node.js", "HTML/CSS"],
        "academic_requirements": {"min_cgpa": 2.8, "min_percentage": 70, "preferred_majors": ["Computer Science", "Software Engineering"]}
    },
    "Mobile Development": {
        "description": "Develop cross-platform mobile applications",
        "required_skills": ["React Native", "Flutter", "Mobile UI/UX"],
        "academic_requirements": {"min_cgpa": 3.0, "min_percentage": 72, "preferred_majors": ["Computer Science", "Mobile Development"]}
    },
    "Cybersecurity": {
        "description": "Protect systems and networks from digital attacks",
        "required_skills": ["Network Security", "Ethical Hacking", "Linux"],
        "academic_requirements": {"min_cgpa": 3.2, "min_percentage": 78, "preferred_majors": ["Cybersecurity", "Computer Science", "IT"]}
    },
    "Cloud Computing": {
        "description": "Design and manage cloud infrastructure",
        "required_skills": ["AWS", "Docker", "Kubernetes"],
        "academic_requirements": {"min_cgpa": 3.0, "min_percentage": 73, "preferred_majors": ["Computer Science", "Cloud Computing"]}
    },
    "AI/ML Engineering": {
        "description": "Develop and deploy machine learning models",
        "required_skills": ["Python", "TensorFlow", "Deep Learning"],
        "academic_requirements": {"min_cgpa": 3.5, "min_percentage": 80, "preferred_majors": ["Computer Science", "AI/ML", "Data Science"]}
    },
    "DevOps": {
        "description": "Streamline development and operations processes",
        "required_skills": ["Docker", "Kubernetes", "CI/CD"],
        "academic_requirements": {"min_cgpa": 3.0, "min_percentage": 75, "preferred_majors": ["Computer Science", "IT"]}
    },
    "Data Engineering": {
        "description": "Design and build data pipelines and infrastructure",
        "required_skills": ["Python", "SQL", "Big Data"],
        "academic_requirements": {"min_cgpa": 3.2, "min_percentage": 76, "preferred_majors": ["Computer Science", "Data Engineering"]}
    }
}

# Company database with additional details
COMPANIES = [
    {
        "name": "TechNova Solutions",
        "size": "1001-5000",
        "industry": "Information Technology & Services",
        "founded": 2010,
        "description": "Leading provider of enterprise software solutions and IT consulting services."
    },
    {
        "name": "DataSphere Inc.",
        "size": "5001-10000",
        "industry": "Data & Analytics",
        "founded": 2008,
        "description": "Pioneers in big data analytics and business intelligence solutions."
    },
    {
        "name": "WebCraft Studios",
        "size": "51-200",
        "industry": "Web Development",
        "founded": 2015,
        "description": "Boutique web development agency specializing in custom web applications."
    },
    {
        "name": "CloudForge Technologies",
        "size": "1001-5000",
        "industry": "Cloud Computing",
        "founded": 2012,
        "description": "Enterprise cloud solutions and infrastructure services provider."
    },
    {
        "name": "AI Dynamics",
        "size": "201-500",
        "industry": "Artificial Intelligence",
        "founded": 2017,
        "description": "Cutting-edge AI research and applied machine learning solutions."
    },
    {
        "name": "SecureNet Systems",
        "size": "1001-5000",
        "industry": "Cybersecurity",
        "founded": 2009,
        "description": "Global leader in enterprise cybersecurity solutions and services."
    },
    {
        "name": "Quantum Innovations",
        "size": "51-200",
        "industry": "Quantum Computing",
        "founded": 2018,
        "description": "Pioneering quantum computing research and development."
    },
    {
        "name": "ByteCraft Labs",
        "size": "11-50",
        "industry": "Software Development",
        "founded": 2019,
        "description": "Innovative software solutions for modern business challenges."
    },
    {
        "name": "NeuralMind AI",
        "size": "201-500",
        "industry": "Artificial Intelligence",
        "founded": 2016,
        "description": "Building the next generation of AI-powered applications."
    },
    {
        "name": "CodeCrafters",
        "size": "51-200",
        "industry": "Software Development",
        "founded": 2014,
        "description": "Custom software development and technical consulting services."
    },
    {
        "name": "DataPulse Analytics",
        "size": "501-1000",
        "industry": "Data Science",
        "founded": 2013,
        "description": "Transforming data into actionable insights for businesses worldwide."
    },
    {
        "name": "CyberShield Security",
        "size": "1001-5000",
        "industry": "Cybersecurity",
        "founded": 2011,
        "description": "Comprehensive cybersecurity solutions for the digital age."
    },
    {
        "name": "AppVantage Mobile",
        "size": "51-200",
        "industry": "Mobile Development",
        "founded": 2016,
        "description": "Creating innovative mobile experiences for global brands."
    },
    {
        "name": "CloudNova Services",
        "size": "5001-10000",
        "industry": "Cloud Computing",
        "founded": 2010,
        "description": "Leading provider of cloud infrastructure and managed services."
    },
    {
        "name": "DeepLearn AI",
        "size": "201-500",
        "industry": "Artificial Intelligence",
        "founded": 2017,
        "description": "Democratizing AI through accessible and scalable machine learning solutions."
    }
]

# Locations with additional metadata
LOCATIONS = [
    {"city": "New York", "state": "NY", "country": "USA", "type": "On-site"},
    {"city": "San Francisco", "state": "CA", "country": "USA", "type": "On-site"},
    {"city": "Austin", "state": "TX", "country": "USA", "type": "Hybrid"},
    {"city": "Seattle", "state": "WA", "country": "USA", "type": "Hybrid"},
    {"city": "Boston", "state": "MA", "country": "USA", "type": "On-site"},
    {"city": "Chicago", "state": "IL", "country": "USA", "type": "On-site"},
    {"city": "Denver", "state": "CO", "country": "USA", "type": "Hybrid"},
    {"city": "Atlanta", "state": "GA", "country": "USA", "type": "On-site"},
    {"city": "Remote", "state": "", "country": "Any", "type": "Remote"},
    {"city": "Bangalore", "state": "Karnataka", "country": "India", "type": "Hybrid"},
    {"city": "London", "state": "", "country": "UK", "type": "On-site"},
    {"city": "Berlin", "state": "", "country": "Germany", "type": "Hybrid"},
    {"city": "Toronto", "state": "ON", "country": "Canada", "type": "On-site"},
    {"city": "Sydney", "state": "NSW", "country": "Australia", "type": "Hybrid"},
    {"city": "Singapore", "state": "", "country": "Singapore", "type": "On-site"}
]

# Internship types with descriptions
INTERNSHIP_TYPES = [
    {"name": "Summer Internship", "duration_months": 3, "start_month": "May", "end_month": "August"},
    {"name": "Fall Internship", "duration_months": 4, "start_month": "September", "end_month": "December"},
    {"name": "Spring Internship", "duration_months": 4, "start_month": "January", "end_month": "April"},
    {"name": "Winter Internship", "duration_months": 1, "start_month": "December", "end_month": "January"},
    {"name": "Co-op", "duration_months": 6, "start_month": "Flexible", "end_month": "Flexible"},
    {"name": "Year-round", "duration_months": 12, "start_month": "Flexible", "end_month": "Flexible"}
]

# Stipend ranges based on location and role level
STIPEND_RANGES = {
    "USA": {
        "entry_level": (2000, 3500),
        "mid_level": (3000, 5000),
        "senior_level": (5000, 8000)
    },
    "Europe": {
        "entry_level": (1500, 2500),
        "mid_level": (2000, 3500),
        "senior_level": (3000, 5000)
    },
    "Asia": {
        "entry_level": (800, 1500),
        "mid_level": (1200, 2500),
        "senior_level": (2000, 4000)
    },
    "Remote": {
        "entry_level": (1000, 2000),
        "mid_level": (1500, 3000),
        "senior_level": (2500, 5000)
    }
}

# Application statuses with additional metadata
APPLICATION_STATUS = [
    {"status": "Open", "accepting_applications": True, "priority": 1},
    {"status": "Early Application", "accepting_applications": True, "priority": 1},
    {"status": "Rolling Basis", "accepting_applications": True, "priority": 2},
    {"status": "Application Review", "accepting_applications": False, "priority": 3},
    {"status": "Waitlist", "accepting_applications": False, "priority": 4},
    {"status": "Closed", "accepting_applications": False, "priority": 5}
]

# Academic requirements
ACADEMIC_LEVELS = ["Undergraduate", "Graduate", "PhD", "Recent Graduate"]

# Benefits and perks
BENEFITS = [
    "Competitive Stipend", "Housing Stipend", "Relocation Assistance", 
    "Health Insurance", "Paid Time Off", "Flexible Hours", 
    "Mentorship Program", "Networking Events", "Wellness Programs",
    "Gym Membership", "Free Meals", "Transportation Allowance"
]

# Application requirements
APPLICATION_REQUIREMENTS = [
    "Resume/CV", "Cover Letter", "Transcript", "Portfolio",
    "Letters of Recommendation", "Coding Challenge", "Video Interview",
    "Technical Assessment", "Case Study", "GitHub Profile"
]

def generate_dummy_internships(num_internships: int = 200) -> List[Dict[str, Any]]:
    """Generate a comprehensive list of dummy internship recommendations.
    
    Args:
        num_internships: Number of internships to generate
        
    Returns:
        List of dictionaries, each representing an internship with detailed attributes
    """
    internships = []
    
    for i in range(1, num_internships + 1):
        # Select a random domain with its metadata
        domain = random.choice(list(DOMAINS.keys()))
        domain_info = DOMAINS[domain]
        
        # Select a company with its details
        company = random.choice(COMPANIES)
        
        # Select a location and determine stipend range based on location
        location = random.choice(LOCATIONS)
        region = "USA" if location["country"] == "USA" else ("Europe" if location["country"] in ["UK", "Germany"] else "Asia")
        region = "Remote" if location["type"] == "Remote" else region
        
        # Determine seniority and corresponding stipend range
        seniority = random.choices(
            ["Entry-level", "Mid-level", "Senior-level"],
            weights=[0.5, 0.35, 0.15],
            k=1
        )[0]
        
        # Get appropriate stipend range based on region and seniority
        stipend_min, stipend_max = STIPEND_RANGES[region][
            "entry_level" if seniority == "Entry-level" else 
            "mid_level" if seniority == "Mid-level" else "senior_level"
        ]
        
        # Add some randomness to the stipend
        stipend = random.randint(
            int(stipend_min * 0.9),  # 10% below min
            int(stipend_max * 1.1)   # 10% above max
        )
        
        # Select skills with varying proficiency requirements
        num_required_skills = random.randint(3, 8)
        skills_required = []
        
        # Ensure we include some domain-required skills
        required_skills = domain_info["required_skills"]
        num_required = min(2, len(required_skills))  # Take 1-2 required skills
        skills_required.extend(random.sample(required_skills, num_required))
        
        # Add additional skills from the domain
        domain_skills = []
        for skill_category, skills in SKILLS_DATABASE.items():
            for skill, details in skills.items():
                if domain.lower() in skill_category.lower() or domain.lower() in [s.lower() for s in details.get("category", "").split(",")]:
                    domain_skills.append((skill, details["level"]))
        
        # Add 1-3 domain-specific skills with appropriate levels
        if domain_skills:
            num_domain_skills = min(random.randint(1, 3), len(domain_skills))
            selected_domain_skills = random.sample(domain_skills, num_domain_skills)
            skills_required.extend([{"name": skill, "level": level} for skill, level in selected_domain_skills])
        
        # Add some general/soft skills
        soft_skills = ["Communication", "Teamwork", "Problem Solving", "Time Management", "Leadership"]
        if random.random() > 0.3:  # 70% chance to add a soft skill
            skills_required.append({"name": random.choice(soft_skills), "level": "Any"})
        
        # Generate a title with seniority and domain
        title_parts = []
        if seniority != "Entry-level":
            title_parts.append(seniority)
        
        # Add domain-specific title components
        domain_titles = {
            "Data Science": ["Data Science", "Machine Learning", "Data Analysis"],
            "Web Development": ["Frontend", "Backend", "Full Stack"],
            "Mobile Development": ["iOS", "Android", "Cross-platform"],
            "Cybersecurity": ["Security", "Penetration Testing", "Threat Intelligence"],
            "Cloud Computing": ["Cloud", "DevOps", "Cloud Infrastructure"],
            "AI/ML Engineering": ["AI", "Machine Learning", "Deep Learning"],
            "DevOps": ["DevOps", "Site Reliability", "Platform Engineering"],
            "Data Engineering": ["Data Engineering", "Big Data", "Data Pipeline"]
        }
        
        title_parts.append(random.choice(domain_titles.get(domain, [domain])))
        title = f"{' '.join(title_parts)} Intern"
        
        # Generate a detailed description
        description = (
            f"Join {company['name']} as a {title} and work on cutting-edge {domain.lower()} projects. "
            f"This {random.choice(['exciting', 'innovative', 'challenging'])} internship offers hands-on experience with "
            f"{', '.join([s['name'] if isinstance(s, dict) else s for s in skills_required[:3]])}, and more. "
            f"Ideal candidates are {random.choice(['passionate', 'driven', 'motivated'])} individuals looking to "
            f"{random.choice(['launch', 'advance', 'elevate'])} their career in {domain}."
        )
        
        # Generate detailed requirements
        requirements = [
            f"Currently pursuing or recently completed a degree in {random.choice(domain_info['academic_requirements']['preferred_majors'])} or related field",
            f"{random.choice(['Basic', 'Strong', 'Solid'])} understanding of {random.choice([s['name'] if isinstance(s, dict) else s for s in skills_required])}",
            f"{random.choice(['Excellent', 'Strong', 'Outstanding'])} problem-solving and analytical skills",
            f"{random.choice(['Effective', 'Exceptional', 'Strong'])} communication and teamwork abilities"
        ]
        
        # Add domain-specific requirements
        if domain == "Data Science":
            requirements.append("Experience with data analysis and visualization tools")
            if random.random() > 0.5:
                requirements.append("Familiarity with statistical modeling and machine learning concepts")
        elif domain == "Web Development":
            requirements.append("Portfolio of web projects (GitHub, personal website, etc.)")
            if random.random() > 0.7:
                requirements.append("Experience with responsive design and cross-browser compatibility")
        elif domain == "Cybersecurity":
            requirements.append("Understanding of security principles and best practices")
            if random.random() > 0.6:
                requirements.append("Familiarity with security tools and frameworks")
        
        # Add some random technical requirements
        if random.random() > 0.5:
            requirements.append("Experience with version control systems (e.g., Git)")
        
        # Randomly add some nice-to-have skills
        if random.random() > 0.7:
            nice_to_have = ["Experience with Agile/Scrum methodologies", 
                          "Previous internship experience in a related field",
                          "Published research or open-source contributions"]
            requirements.append(f"{random.choice(['Bonus:', 'Nice to have:', 'Plus:'])} {random.choice(nice_to_have)}")
        
        # Select internship type and duration
        internship_type = random.choice(INTERNSHIP_TYPES)
        
        # Generate application dates
        posted_date = datetime.now() - timedelta(days=random.randint(1, 60))
        deadline_days = random.randint(7, 90)
        deadline_date = posted_date + timedelta(days=deadline_days)
        
        # Determine application status based on dates
        days_until_deadline = (deadline_date - datetime.now()).days
        if days_until_deadline < 0:
            status = random.choices(
                ["Closed", "Application Review"],
                weights=[0.7, 0.3],
                k=1
            )[0]
        elif days_until_deadline < 7:
            status = random.choices(
                ["Open", "Early Application", "Application Review"],
                weights=[0.4, 0.1, 0.5],
                k=1
            )[0]
        else:
            status = random.choices(
                ["Open", "Early Application", "Rolling Basis"],
                weights=[0.5, 0.3, 0.2],
                k=1
            )[0]
        
        # Select random benefits (2-5 benefits)
        num_benefits = random.randint(2, 5)
        benefits = random.sample(BENEFITS, num_benefits)
        
        # Select application requirements (2-4 requirements)
        num_app_reqs = random.randint(2, 4)
        application_requirements = random.sample(APPLICATION_REQUIREMENTS, num_app_reqs)
        
        # Create the internship dictionary with all details
        internship = {
            "id": str(uuid.uuid4())[:8],  # Generate a short unique ID
            "title": title,
            "company": company["name"],
            "company_details": {
                "size": company["size"],
                "industry": company["industry"],
                "founded": company["founded"],
                "description": company["description"]
            },
            "location": {
                "city": location["city"],
                "state": location["state"],
                "country": location["country"],
                "type": location["type"],
                "is_remote": location["type"] == "Remote"
            },
            "internship_details": {
                "type": internship_type["name"],
                "duration_months": internship_type["duration_months"],
                "start_date": (datetime.now() + timedelta(days=random.randint(30, 120))).strftime("%Y-%m-%d"),
                "end_date": (datetime.now() + timedelta(days=random.randint(120, 365))).strftime("%Y-%m-%d"),
                "is_paid": random.random() > 0.1,  # 90% chance of being paid
                "stipend_amount": stipend,
                "stipend_currency": "USD",
                "stipend_frequency": "monthly"
            },
            "skills_required": skills_required,
            "description": description,
            "responsibilities": [
                f"Assist with {random.choice(['developing', 'designing', 'implementing'])} {random.choice(['new features', 'solutions', 'systems'])} in {domain}",
                f"Collaborate with {random.choice(['cross-functional', 'distributed'])} teams to {random.choice(['solve', 'address', 'tackle'])} complex problems",
                f"{random.choice(['Research', 'Investigate', 'Explore'])} {random.choice(['emerging', 'new', 'cutting-edge'])} technologies and {random.choice(['methodologies', 'approaches', 'techniques'])}",
                f"{random.choice(['Document', 'Present', 'Report on'])} {random.choice(['findings', 'results', 'progress'])} to {random.choice(['team members', 'stakeholders', 'leadership'])}"
            ],
            "requirements": requirements,
            "academic_requirements": {
                "min_cgpa": domain_info["academic_requirements"]["min_cgpa"],
                "min_percentage": domain_info["academic_requirements"]["min_percentage"],
                "preferred_majors": domain_info["academic_requirements"]["preferred_majors"],
                "academic_level": random.choices(
                    ACADEMIC_LEVELS,
                    weights=[0.5, 0.3, 0.1, 0.1],  # Higher weight for undergrads
                    k=1
                )[0]
            },
            "application_details": {
                "status": status,
                "is_active": status in ["Open", "Early Application", "Rolling Basis"],
                "posted_date": posted_date.strftime("%Y-%m-%d"),
                "deadline": deadline_date.strftime("%Y-%m-%d"),
                "requirements": application_requirements,
                "process": [
                    "Submit application",
                    "Technical assessment" if random.random() > 0.3 else "Initial screening",
                    "Technical interview" if random.random() > 0.5 else "Behavioral interview",
                    "Final interview" if random.random() > 0.7 else "Team interview",
                    "Offer"
                ]
            },
            "benefits": benefits,
            "domain": domain,
            "domain_category": domain_info.get("category", "General"),
            "seniority_level": seniority,
            "company_logo": f"https://logo.clearbit.com/{company['name'].lower().replace(' ', '')}.com",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "applications_received": random.randint(5, 500),
                "views": random.randint(50, 5000)
            }
        }
        
        # Add some additional metadata for search and filtering
        if random.random() > 0.7:
            internship["is_featured"] = True
            internship["is_urgent"] = random.random() > 0.7
        
        if random.random() > 0.8:
            internship["sponsorship_available"] = random.random() > 0.5
        
        internships.append(internship)
    
    return internships

def save_to_json(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Save the generated data to a JSON file with proper formatting.
    
    Args:
        data: The data to save
        filepath: Path to the output JSON file
    """
    # Create the directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate some statistics
    domains = {}
    companies = {}
    locations = {}
    
    for item in data:
        # Count domains
        domain = item.get('domain', 'Unknown')
        domains[domain] = domains.get(domain, 0) + 1
        
        # Count companies
        company = item.get('company', 'Unknown')
        companies[company] = companies.get(company, 0) + 1
        
        # Count locations
        loc = item.get('location', {})
        loc_str = f"{loc.get('city', '')}, {loc.get('country', '')}"
        locations[loc_str] = locations.get(loc_str, 0) + 1
    
    # Create a summary
    summary = {
        "metadata": {
            "total_internships": len(data),
            "generated_at": datetime.now().isoformat(),
            "domains": domains,
            "companies": companies,
            "locations": locations,
            "date_range": {
                "earliest_start_date": min(item['internship_details']['start_date'] for item in data),
                "latest_end_date": max(item['internship_details']['end_date'] for item in data)
            }
        },
        "internships": data
    }
    
    # Save to JSON with indentation for readability
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n[SUCCESS] Successfully generated {len(data)} dummy internships at {filepath}")
    print(f"   - Domains: {', '.join(f'{k} ({v})' for k, v in domains.items())}")
    print(f"   - Companies: {len(companies)} unique companies")
    print(f"   - Locations: {len(locations)} unique locations")
    print(f"   - Timeframe: {summary['metadata']['date_range']['earliest_start_date']} to {summary['metadata']['date_range']['latest_end_date']}")
    print("\n=== Dataset is ready for testing!")
    
    # Save a smaller sample for quick testing
    sample_size = min(20, len(data) // 5)
    sample_file = filepath.parent / "sample_internships.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump({"internships": data[:sample_size]}, f, indent=2, ensure_ascii=False)
    
    print(f"=== Sample dataset with {sample_size} internships saved to {sample_file}")

def main():
    """Generate and save the dummy internship recommendations."""
    print("=== Starting dummy data generation...")
    print("=== Creating comprehensive internship dataset...")
    
    try:
        # Generate 200 dummy internships for a robust test set
        num_internships = 200
        print(f"=== Generating {num_internships} diverse internship listings...")
        internships = generate_dummy_internships(num_internships)
        
        # Save to JSON file
        print("=== Saving data to file...")
        save_to_json(internships, OUTPUT_FILE)
        
    except Exception as e:
        print(f"\n=== Error generating dummy data: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("=== Internship Recommendation System - Dummy Data Generator ===")
    print("=" * 70)
    print("This script generates a comprehensive dataset of internship listings")
    print("with varied requirements, skills, and attributes for testing purposes.\n")
    
    start_time = time.time()
    return_code = main()
    end_time = time.time()
    
    print("\n" + "=" * 70)
    print("=== Dummy data generation complete!")
    print(f"=== Time taken: {end_time - start_time:.2f} seconds")
    print("=" * 70)
    
    if return_code == 0:
        print("\nNext steps:")
        print(f"1. The dataset is available at: {OUTPUT_FILE}")
        print("2. Use the recommendation engine to test different CVs")
        print("3. Check the sample dataset for a smaller test set\n")
    
    exit(return_code)
