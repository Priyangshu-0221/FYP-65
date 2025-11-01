"""Run the FastAPI application."""
import uvicorn
from backend.fastapi_app import app

if __name__ == "__main__":
    uvicorn.run("backend.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
