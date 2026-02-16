"""Main entry point for the FastAPI application."""

import uvicorn
from .app import app

if __name__ == "__main__":
    print("Starting Simple Resume Processor API...")
    print("API will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
