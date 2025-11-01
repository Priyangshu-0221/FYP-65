"""Main entry point for the FastAPI application."""

import uvicorn
from fastapi_app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
