"""Utility functions for model persistence."""

import json
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

# Model directory and file paths
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "resume_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def save_model(
    model: BaseEstimator,
    vectorizer: TfidfVectorizer,
    metrics: Optional[Dict[str, Any]] = None,
    model_type: str = "logistic_regression",
    data_path: Optional[Path] = None,
) -> None:
    """
    Save the trained model, vectorizer, and metadata to disk.
    
    Args:
        model: Trained scikit-learn model
        vectorizer: Fitted TfidfVectorizer
        metrics: Dictionary of model metrics
        model_type: Type of the model
        data_path: Path to the training data
    """
    # Create model directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "data_path": str(data_path) if data_path else None,
        "metrics": metrics or {},
        "vectorizer_params": vectorizer.get_params()
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_model() -> Tuple[Optional[BaseEstimator], Optional[TfidfVectorizer], dict]:
    """
    Load the trained model, vectorizer, and metadata from disk.
    
    Returns:
        Tuple of (model, vectorizer, metadata) if found, (None, None, {}) otherwise
    """
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists() or not METADATA_PATH.exists():
        return None, None, {}
    
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            
        return model, vectorizer, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, {}


def model_exists() -> bool:
    """Check if a trained model exists on disk."""
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists() and METADATA_PATH.exists()
