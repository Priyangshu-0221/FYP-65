"""
Simple Backend API for Resume Upload
Uses the simple_extractor.py for PDF processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import sys
from collections import Counter

# Import from same package
from .extractor import process_resume
from .recommendation_engine import InternshipRecommendationEngine, AVAILABLE_MODELS

app = FastAPI(title="AI Career Guidance System")

# Initialize recommendation engine
# Default to bge-small for better research-backed accuracy
try:
    engine = InternshipRecommendationEngine(model_name=AVAILABLE_MODELS["bge-small"])
except Exception as e:
    print(f"Warning: Could not load advanced engine: {e}. Falling back to basic matching.")
    engine = None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}


@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and process a resume PDF
    Returns: extracted skills, education, experience, and academic scores
    """
    print(f"\n=== Upload Request ===")
    print(f"Filename: {file.filename}")
    print(f"Content-Type: {file.content_type}")
    
    # Check if it's a PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        print(f"File size: {len(content)} bytes")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Process the PDF
            print(f"Processing PDF: {temp_path}")
            result = process_resume(temp_path, category="Uploaded")
            
            if not result:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to extract information from PDF"
                )
            
            print(f"Successfully extracted {result['skills_count']} skills")
            
            # Return response
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "text": result['text'][:1000] + '...' if len(result['text']) > 1000 else result['text'],
                    "skills": result['skills'],
                    "category": "General",
                    "education": result['education'],
                    "experience": result['experience'],
                    "skill_count": result['skills_count'],
                    "cgpa": result['cgpa'],
                    "percentage": result['percentage']
                }
            )
            
        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except:
                pass
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )


@app.post("/recommend")
async def recommend_internships(payload: dict):
    """
    Get internship recommendations based on skills with scoring for better relevance.
    """
    user_skills = [s.lower() for s in payload.get('skills', [])]
    top_k = payload.get('top_k', 6)
    
    if not user_skills:
        raise HTTPException(status_code=400, detail="No skills provided")
    
    # Expanded Dummy Internship Database
    # Use advanced engine if available
    if engine:
        try:
            # Prepare marks for engine
            marks = payload.get('marks', {})
            
            # Get recommendations from engine
            recommended_jobs = engine.recommend_internships(
                user_skills=user_skills,
                user_marks=marks,
                top_n=top_k
            )
            
            # Get recommended skills from engine
            recommended_skills = engine.get_skill_recommendations(
                user_skills=user_skills,
                recommendations=recommended_jobs
            )
            
            if recommended_jobs:
                return {
                    "recommendations": recommended_jobs,
                    "recommended_skills": recommended_skills
                }
        except Exception as e:
            print(f"Engine recommendation failed: {e}. Falling back to basic matching.")

    # Fallback to basic Matching Logic (Original Logic)
    all_internships = [
        # --- TECHNICAL ROLES ---
        {"id": "1", "title": "Software Developer Intern", "company": "Tech Innovations", "location": "Bangalore, India", "category": "Software", "skills": ["python", "java", "sql", "git"], "description": "Build scalable apps using Python and Java.", "apply_link": "#"},
        {"id": "2", "title": "Frontend Intern", "company": "WebCraft", "location": "Pune, India", "category": "Web", "skills": ["react", "javascript", "html", "css"], "description": "Create amazing user experiences with React.", "apply_link": "#"},
        {"id": "3", "title": "Data Analyst Intern", "company": "Insight Data", "location": "Mumbai, India", "category": "Data", "skills": ["python", "sql", "pandas", "tableau"], "description": "Turn complex data into actionable insights.", "apply_link": "#"},
        {"id": "4", "title": "Full Stack Intern", "company": "MERN Systems", "location": "Remote, India", "category": "Web", "skills": ["mongodb", "express", "react", "node.js"], "description": "Develop end-to-end web solutions remotely.", "apply_link": "#"},
        {"id": "5", "title": "AI/ML Intern", "company": "Neural Hub", "location": "Hyderabad, India", "category": "AI", "skills": ["python", "tensorflow", "pytorch", "scikit-learn"], "description": "Assist in building next-gen ML models.", "apply_link": "#"},
        {"id": "6", "title": "DevOps Intern", "company": "CloudScale", "location": "Chennai, India", "category": "Cloud", "skills": ["docker", "kubernetes", "aws", "linux"], "description": "Automate deployment pipelines and cloud infrastructure.", "apply_link": "#"},
        {"id": "7", "title": "Cybersecurity Intern", "company": "GuardNet", "location": "Delhi NCR, India", "category": "Security", "skills": ["linux", "security", "networking", "python"], "description": "Protect systems from cyber threats.", "apply_link": "#"},
        {"id": "8", "title": "Mobile Developer Intern", "company": "AppFlow", "location": "Ahmedabad, India", "category": "Mobile", "skills": ["swift", "kotlin", "flutter", "dart"], "description": "Build native and cross-platform mobile apps.", "apply_link": "#"},
        
        # --- FINANCE & ACCOUNTING ---
        {"id": "9", "title": "Tax Accountant Intern", "company": "Finance Pros", "location": "Kolkata, India", "category": "Finance", "skills": ["accounting", "tax", "excel", "finance"], "description": "Support tax preparation and financial reporting.", "apply_link": "#"},
        {"id": "10", "title": "Investment Analyst Intern", "company": "Peak Capital", "location": "London, UK", "category": "Finance", "skills": ["economics", "finance", "excel", "analytical"], "description": "Analyze market trends and investment opportunities.", "apply_link": "#"},
        {"id": "11", "title": "Risk Management Intern", "company": "SafeGuard Bank", "location": "Singapore", "category": "Finance", "skills": ["risk", "compliance", "finance", "audit"], "description": "Identify and mitigate financial risks.", "apply_link": "#"},

        # --- HR & MANAGEMENT ---
        {"id": "12", "title": "HR Specialist Intern", "company": "People Power", "location": "Bangalore, India", "category": "HR", "skills": ["recruitment", "interviewing", "sourcing", "communication"], "description": "Help find and hire top talent.", "apply_link": "#"},
        {"id": "13", "title": "Training Coordinator Intern", "company": "SkillUp Corp", "location": "Remote", "category": "HR", "skills": ["training", "presentation", "leadership", "organization"], "description": "Organize employee development programs.", "apply_link": "#"},
        {"id": "14", "title": "Business Development Intern", "company": "Growth Engine", "location": "Gurgaon, India", "category": "Sales", "skills": ["sales", "negotiation", "crm", "marketing"], "description": "Assist in sales outreach and market expansion.", "apply_link": "#"},

        # --- MARKETING & DESIGN ---
        {"id": "15", "title": "Digital Marketer Intern", "company": "Social Buzz", "location": "Remote, India", "category": "Marketing", "skills": ["seo", "social media", "content", "analytics"], "description": "Manage online marketing campaigns.", "apply_link": "#"},
        {"id": "16", "title": "UX/UI Designer Intern", "company": "Pixel Perfect", "location": "Dubai, UAE", "category": "Design", "skills": ["figma", "sketch", "adobe xd", "prototyping"], "description": "Design user-centric web & mobile interfaces.", "apply_link": "#"},
        {"id": "17", "title": "Graphic Design Intern", "company": "Creative Hub", "location": "Mumbai, India", "category": "Design", "skills": ["photoshop", "illustrator", "branding", "layout"], "description": "Create stunning visual assets for brands.", "apply_link": "#"},
        {"id": "18", "title": "Public Relations Intern", "company": "Global PR", "location": "New York, NY", "category": "PR", "skills": ["writing", "media", "networking", "press"], "description": "Assist in drafting press releases and media outreach.", "apply_link": "#"},

        # --- HEALTHCARE & SCIENCE ---
        {"id": "19", "title": "Healthcare Admin Intern", "company": "City Hospital", "location": "Delhi, India", "category": "Healthcare", "skills": ["management", "healthcare", "hipaa", "organization"], "description": "Support day-to-day hospital operations.", "apply_link": "#"},
        {"id": "20", "title": "Lab Research Intern", "company": "BioScience Lab", "location": "San Diego, CA", "category": "Science", "skills": ["biology", "chemistry", "research", "documentation"], "description": "Conduct laboratory experiments and data tracking.", "apply_link": "#"},
        {"id": "21", "title": "Public Health Intern", "company": "Health First", "location": "Remote", "category": "Healthcare", "skills": ["epidemiology", "health", "advocacy", "research"], "description": "Work on community health outreach programs.", "apply_link": "#"},

        # --- ENGINEERING ---
        {"id": "22", "title": "Civil Engineer Intern", "company": "BuildRight", "location": "Pune, India", "category": "Engineering", "skills": ["autocad", "math", "construction", "design"], "description": "Assist in structural design and blueprint mapping.", "apply_link": "#"},
        {"id": "23", "title": "Mechanical Intern", "company": "Apex Motors", "location": "Chennai, India", "category": "Engineering", "skills": ["solidworks", "cad", "mechanics", "manufacturing"], "description": "Support machine testing and part design.", "apply_link": "#"},
        {"id": "24", "title": "Electrical Intern", "company": "PowerGrid", "location": "Bangalore, India", "category": "Engineering", "skills": ["circuitry", "matlab", "embedded", "testing"], "description": "Help design and test electrical components.", "apply_link": "#"},

        # --- LEGAL & EDUCATION ---
        {"id": "25", "title": "Legal Assistant Intern", "company": "Lex Chambers", "location": "Remote, India", "category": "Legal", "skills": ["research", "writing", "law", "contracts"], "description": "Assist lawyers with case research and paperwork.", "apply_link": "#"},
        {"id": "26", "title": "Elementary Tutor Intern", "company": "Bright Kids", "location": "Remote", "category": "Education", "skills": ["teaching", "patience", "writing", "english"], "description": "Support young learners in core subjects.", "apply_link": "#"},
        {"id": "27", "title": "Content Writer Intern", "company": "Edit Masters", "location": "Remote", "category": "Writing", "skills": ["blogging", "editing", "seo", "storytelling"], "description": "Write engaging articles and website content.", "apply_link": "#"},

        # --- ADDITIONAL DIVERSE ROLES ---
        {"id": "30", "title": "Agri-Tech Intern", "company": "FarmSense", "location": "Jaipur, India", "category": "Agriculture", "skills": ["python", "sensors", "data", "farming"], "description": "Use tech to optimize crop yields.", "apply_link": "#"},
        {"id": "31", "title": "Hospitality Intern", "company": "Star Resorts", "location": "Goa, India", "category": "Hospitality", "skills": ["management", "service", "tourism", "organization"], "description": "Learn luxury hotel operations.", "apply_link": "#"},
        {"id": "32", "title": "Supply Chain Intern", "company": "LogiRoute", "location": "Nagpur, India", "category": "Logistics", "skills": ["inventory", "planning", "excel", "purchasing"], "description": "Coordinate shipping and inventory management.", "apply_link": "#"},
        {"id": "33", "title": "Sustainable Energy Intern", "company": "EcoPower", "location": "Berlin, Germany", "category": "Energy", "skills": ["energy", "policy", "solar", "wind"], "description": "Research renewable energy solutions.", "apply_link": "#"},
        {"id": "34", "title": "Fashion Intern", "company": "StyleHouse", "location": "Milan, IT", "category": "Fashion", "skills": ["trends", "marketing", "buying", "research"], "description": "Assist in seasonal trend analysis.", "apply_link": "#"},
        {"id": "35", "title": "Social Impact Intern", "company": "Green Future", "location": "Remote", "category": "NGO", "skills": ["advocacy", "writing", "non-profit", "fundraising"], "description": "Coordinate donation and impact campaigns.", "apply_link": "#"}
    ]
    
    results = []
    import random
    
    for job in all_internships:
        job_skills = [s.lower() for s in job['skills']]
        matches = list(set(user_skills) & set(job_skills))
        score = len(matches)
        final_score = score + (random.random() * 0.1)
        
        if score > 0:
            results.append({
                "internship": job,
                "score": final_score,
                "match_count": score
            })
            
    results.sort(key=lambda x: x['score'], reverse=True)
    recommended_jobs = [r['internship'] for r in results[:top_k]]
    
    if not recommended_jobs:
        recommended_jobs = random.sample(all_internships, k=min(top_k, len(all_internships)))
    
    # Simple skill recommendation for fallback
    missing_skills_counter = Counter()
    for job in recommended_jobs:
        job_skills = [s.lower() for s in job.get('skills', [])]
        for skill in job_skills:
            if skill not in user_skills:
                missing_skills_counter[skill.capitalize()] += 1
                
    recommended_skills = [skill for skill, _ in missing_skills_counter.most_common(15)]
    
    return {
        "recommendations": recommended_jobs,
        "recommended_skills": recommended_skills
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting AI Career Guidance System...")
    print("API will be available at: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
