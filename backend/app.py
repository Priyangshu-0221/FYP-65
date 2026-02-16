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

# Import from same package
from .extractor import process_resume

app = FastAPI(title="Simple Resume Processor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
            
            print(f"✓ Extracted {result['skills_count']} skills")
            
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
    Get internship recommendations based on skills
    """
    skills = payload.get('skills', [])
    top_k = payload.get('top_k', 6)
    
    if not skills:
        raise HTTPException(status_code=400, detail="No skills provided")
    
    # Dummy internship database
    all_internships = [
        {
            "id": "1",
            "title": "Software Developer Intern",
            "company": "Tech Innovations Inc",
            "location": "Remote",
            "category": "Software Development",
            "skills": ["python", "java", "sql", "git"],
            "description": "Join our team to build scalable web applications using modern technologies. Work on real-world projects and learn from experienced developers.",
            "apply_link": "https://example.com/apply/1"
        },
        {
            "id": "2",
            "title": "Frontend Developer Intern",
            "company": "WebCraft Studios",
            "location": "San Francisco, CA",
            "category": "Web Development",
            "skills": ["react", "javascript", "html", "css", "typescript"],
            "description": "Create beautiful, responsive user interfaces for our web applications. Learn modern frontend frameworks and best practices.",
            "apply_link": "https://example.com/apply/2"
        },
        {
            "id": "3",
            "title": "Data Science Intern",
            "company": "DataMinds Analytics",
            "location": "New York, NY",
            "category": "Data Science",
            "skills": ["python", "machine learning", "pandas", "numpy", "sql"],
            "description": "Analyze large datasets and build predictive models. Gain hands-on experience with ML algorithms and data visualization.",
            "apply_link": "https://example.com/apply/3"
        },
        {
            "id": "4",
            "title": "Mobile App Developer Intern",
            "company": "AppVenture Labs",
            "location": "Austin, TX",
            "category": "Mobile Development",
            "skills": ["react native", "javascript", "android", "ios"],
            "description": "Develop cross-platform mobile applications for iOS and Android. Work with cutting-edge mobile technologies.",
            "apply_link": "https://example.com/apply/4"
        },
        {
            "id": "5",
            "title": "DevOps Engineer Intern",
            "company": "CloudOps Solutions",
            "location": "Seattle, WA",
            "category": "DevOps",
            "skills": ["docker", "kubernetes", "aws", "linux", "ci/cd"],
            "description": "Learn cloud infrastructure and automation. Help maintain and improve our deployment pipelines.",
            "apply_link": "https://example.com/apply/5"
        },
        {
            "id": "6",
            "title": "Backend Developer Intern",
            "company": "ServerSide Tech",
            "location": "Remote",
            "category": "Backend Development",
            "skills": ["node.js", "python", "django", "postgresql", "rest api"],
            "description": "Build robust backend systems and APIs. Work with databases, authentication, and server-side logic.",
            "apply_link": "https://example.com/apply/6"
        },
        {
            "id": "7",
            "title": "Machine Learning Intern",
            "company": "AI Research Labs",
            "location": "Boston, MA",
            "category": "Artificial Intelligence",
            "skills": ["python", "tensorflow", "pytorch", "deep learning", "nlp"],
            "description": "Research and implement ML models for real-world applications. Work on NLP, computer vision, and more.",
            "apply_link": "https://example.com/apply/7"
        },
        {
            "id": "8",
            "title": "Full Stack Developer Intern",
            "company": "FullStack Ventures",
            "location": "Chicago, IL",
            "category": "Full Stack Development",
            "skills": ["react", "node.js", "mongodb", "express", "javascript"],
            "description": "Work on both frontend and backend of modern web applications. MERN stack experience preferred.",
            "apply_link": "https://example.com/apply/8"
        },
        {
            "id": "9",
            "title": "Cybersecurity Intern",
            "company": "SecureNet Systems",
            "location": "Washington, DC",
            "category": "Cybersecurity",
            "skills": ["network security", "python", "linux", "penetration testing"],
            "description": "Learn about security vulnerabilities and how to protect systems. Conduct security audits and assessments.",
            "apply_link": "https://example.com/apply/9"
        },
        {
            "id": "10",
            "title": "UI/UX Design Intern",
            "company": "DesignFirst Agency",
            "location": "Los Angeles, CA",
            "category": "Design",
            "skills": ["figma", "adobe xd", "ui design", "ux research", "prototyping"],
            "description": "Create user-centered designs for web and mobile applications. Conduct user research and usability testing.",
            "apply_link": "https://example.com/apply/10"
        },
        {
            "id": "11",
            "title": "Cloud Engineer Intern",
            "company": "CloudScale Technologies",
            "location": "Remote",
            "category": "Cloud Computing",
            "skills": ["aws", "azure", "gcp", "terraform", "python"],
            "description": "Build and manage cloud infrastructure. Learn about scalable architectures and cloud-native applications.",
            "apply_link": "https://example.com/apply/11"
        },
        {
            "id": "12",
            "title": "QA Automation Intern",
            "company": "TestPro Solutions",
            "location": "Denver, CO",
            "category": "Quality Assurance",
            "skills": ["selenium", "python", "testing", "automation", "ci/cd"],
            "description": "Develop automated test suites and ensure software quality. Learn testing frameworks and best practices.",
            "apply_link": "https://example.com/apply/12"
        }
    ]
    
    # Simple matching: return internships that match any of the user's skills
    matched_internships = []
    for internship in all_internships:
        # Check if any user skill matches any internship skill
        skill_match = any(
            user_skill.lower() in [s.lower() for s in internship['skills']]
            for user_skill in skills
        )
        if skill_match:
            matched_internships.append(internship)
    
    # If no matches, return all internships
    if not matched_internships:
        matched_internships = all_internships
    
    # Return top_k internships
    return {
        "recommendations": matched_internships[:top_k]
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Simple Resume Processor API...")
    print("API will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
