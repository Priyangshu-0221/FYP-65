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
    
    if not skills:
        raise HTTPException(status_code=400, detail="No skills provided")
    
    # For now, return a simple response
    # You can integrate with your recommendation engine later
    return {
        "recommendations": [
            {
                "id": "1",
                "title": "Software Developer Intern",
                "company": "Tech Corp",
                "location": "Remote",
                "category": "IT",
                "skills": skills[:3],
                "description": "Great opportunity for developers",
                "apply_link": "#"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Simple Resume Processor API...")
    print("API will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
