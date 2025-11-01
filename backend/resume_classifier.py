"""
Resume Classification Module

This module provides functionality for training and using resume classification models.
It includes text preprocessing, feature extraction, model training, and evaluation.
"""

import argparse
import json
import re
import joblib
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

from backend.skill_extractor import ensure_nltk_dependencies

# Default model type
DEFAULT_MODEL_TYPE = "logistic_regression"

# Model directory and file paths
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "resume_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# =============================================================================
# Model Persistence Utilities
# =============================================================================

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
    if not model_exists():
        return None, None, {}
    
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        metadata = {}
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
        
        return model, vectorizer, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, {}


def model_exists() -> bool:
    """Check if a trained model exists on disk."""
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return " ".join(pages)
    except Exception:
        return ""


def load_resume_dataset(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load resume dataset from a directory structure where:
    DATA/
    └── data/
        ├── ACCOUNTANT/
        │   ├── 10554236.pdf
        │   └── ...
        ├── ADVOCATE/
        │   ├── 10186968.pdf
        │   └── ...
        └── ... (more occupation folders)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    data = []
    processed_files = 0
    
    # Iterate through each occupation folder
    for occupation_dir in sorted(data_dir.iterdir()):
        if not occupation_dir.is_dir() or occupation_dir.name.startswith('.'):
            continue
            
        occupation = occupation_dir.name.replace('-', ' ').title()  # Convert 'ACCOUNTANT' to 'Accountant'
        print(f"Processing {occupation}...")
        
        # Process each PDF in the occupation folder
        pdf_files = list(occupation_dir.glob('*.pdf'))
        if not pdf_files:
            print(f"  No PDF files found in {occupation_dir}")
            continue
            
        for pdf_file in pdf_files:
            try:
                text = extract_text_from_pdf(pdf_file)
                if text and text.strip():
                    data.append({
                        'resume_text': text,
                        'category': occupation,
                        'filename': pdf_file.name,
                        'filepath': str(pdf_file.relative_to(data_dir.parent))
                    })
                    processed_files += 1
                    if processed_files % 50 == 0:
                        print(f"  Processed {processed_files} files...")
            except Exception as e:
                print(f"  Error processing {pdf_file.name}: {str(e)}")
    
    if not data:
        raise ValueError(f"No valid PDF files found in {data_dir}")
        
    print(f"Successfully loaded {len(data)} resumes from {processed_files} files across {len(set(d['category'] for d in data))} categories")
    return pd.DataFrame(data)


def clean_text(text: str) -> str:
    """Clean and normalize text with multiple preprocessing steps."""
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove phone numbers
    text = re.sub(r'\b(?:\+?[0-9]{1,3}[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', ' ', text)
    
    # Remove special characters and numbers but keep common special chars in skills (like C++, .NET)
    text = re.sub(r'[^a-z\s+.#\-]', ' ', text)
    
    # Handle specific tech terms (e.g., C++, C#, .NET)
    text = re.sub(r'\b(c\+\+|c#|\.net|asp\.net|node\.js|react\.js|d3\.js)\b', lambda x: x.group().replace('.', '_'), text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    """
    Enhanced text preprocessing with advanced cleaning, tokenization, and lemmatization.
    
    Args:
        text: Input text to preprocess
        stop_words: Set of stopwords to remove
        lemmatizer: WordNetLemmatizer instance
        
    Returns:
        Preprocessed text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""
        
    try:
        # Clean text
        cleaned = clean_text(text)
        if not cleaned:
            return ""
            
        # Tokenize
        tokens = word_tokenize(cleaned)
        
        # Initialize list for processed tokens
        processed_tokens = []
        
        # Get part-of-speech tags for better lemmatization
        pos_tags = pos_tag(tokens)
        
        for i, (token, pos_tag) in enumerate(zip(tokens, pos_tags)):
            # Skip short tokens and stopwords
            if len(token) <= 1 or token in stop_words:
                continue
                
            # Get POS tag and convert to WordNet format
            pos = get_wordnet_pos(pos_tag[1])
            
            # Handle special cases (e.g., programming languages)
            if '_' in token:  # Handle cases like 'c++' converted to 'c__'
                processed_tokens.append(token.replace('_', ''))
                continue
                
            # Lemmatize with appropriate POS tag
            try:
                if pos:
                    lemma = lemmatizer.lemmatize(token, pos=pos)
                else:
                    lemma = lemmatizer.lemmatize(token)
                    
                # Only add if the lemma is meaningful
                if len(lemma) > 1 and not lemma.isdigit():
                    processed_tokens.append(lemma)
            except:
                # Fallback to original token if lemmatization fails
                processed_tokens.append(token)
        
        return " ".join(processed_tokens)
        
    except Exception as e:
        print(f"Error in preprocessing text: {str(e)}")
        return clean_text(text)  # Fallback to basic cleaning


def get_wordnet_pos(treebank_tag: str) -> str:
    """Convert treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def preprocess_corpus(text_series: pd.Series) -> list[str]:
    ensure_nltk_dependencies()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return [preprocess_text(text, stop_words, lemmatizer) for text in text_series]


def build_vectorizer(max_features: int = 10000, ngram_range=(1, 3), min_df=2, max_df=0.9):
    """Build a TF-IDF vectorizer with enhanced settings.
    
    Args:
        max_features: Maximum number of features
        ngram_range: Range of n-grams to include
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        
    Returns:
        Configured TfidfVectorizer instance
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',
        sublinear_tf=True,  # Use sublinear tf scaling
        norm='l2',  # Normalize vectors
        smooth_idf=True,  # Smooth idf weights
        use_idf=True,  # Enable inverse-document-frequency reweighting
        analyzer='word',  # Feature should be made of word n-grams
        token_pattern=r'\b[a-z][a-z0-9_]+\b'  # Custom token pattern
    )


def vectorize_text(train_texts: list[str], test_texts: list[str], vectorizer: TfidfVectorizer | None = None):
    """
    Vectorize text data using TF-IDF.
    
    Args:
        train_texts: List of training text documents
        test_texts: List of test text documents
        vectorizer: Optional pre-fitted vectorizer
        
    Returns:
        Tuple of (vectorizer, X_train, X_test)
    """
    if vectorizer is None:
        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(train_texts)
    else:
        X_train = vectorizer.transform(train_texts)
    
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def train_model(X_train, y_train, model_type: str = "logistic_regression", cv: bool = True):
    """
    Train a classification model with optional cross-validation.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        model_type: Type of model to train ('naive_bayes' or 'logistic_regression')
        cv: Whether to use cross-validation for hyperparameter tuning
        
    Returns:
        Trained model and training metrics
    """
    if model_type == "naive_bayes":
        if cv:
            # Hyperparameter grid for Naive Bayes
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
                'fit_prior': [True, False]
            }
            grid = GridSearchCV(
                MultinomialNB(),
                param_grid=param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"Best Naive Bayes params: {grid.best_params_}")
        else:
            model = MultinomialNB(alpha=1.0)
    else:  # logistic regression
        if cv:
            # Hyperparameter grid for Logistic Regression
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
                'class_weight': [None, 'balanced']
            }
            grid = GridSearchCV(
                LogisticRegression(max_iter=1000, random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"Best Logistic Regression params: {grid.best_params_}")
        else:
            model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    
    # Train the final model
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder=None):
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        label_encoder: Optional label encoder for class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Detailed classification report
    report = classification_report(
        y_test, 
        y_pred, 
        output_dict=True,
        target_names=label_encoder.classes_ if label_encoder else None
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate top-k accuracy if probabilities are available
    top_k_accuracy = None
    if y_pred_proba is not None and len(np.unique(y_test)) > 2:
        top_k = min(3, len(np.unique(y_test)))
        top_k_accuracy = top_k_accuracy_score(
            y_test, y_pred_proba, k=top_k, labels=np.arange(y_pred_proba.shape[1])
        )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "top_k_accuracy": top_k_accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }


def predict_resume_category(model, vectorizer, resume_text: str) -> str:
    """
    Predict the category of a resume.
    
    Args:
        model: Trained model
        vectorizer: Fitted TfidfVectorizer
        resume_text: Text content of the resume
        
    Returns:
        Predicted category as a string
    """
    if not model or not vectorizer:
        raise ValueError("Model and vectorizer must be provided")
        
    try:
        processed_text = preprocess_corpus(pd.Series([resume_text]))[0]
        features = vectorizer.transform([processed_text])
        return model.predict(features)[0]
    except Exception as e:
        print(f"Error predicting category: {e}")
        return "unknown"


def get_or_train_model(
    data_path: Path,
    model_type: str = DEFAULT_MODEL_TYPE,
    force_retrain: bool = False,
    **train_kwargs
) -> Tuple[BaseEstimator, TfidfVectorizer]:
    """
    Get the trained model and vectorizer, loading from disk if available or training if not.
    
    Args:
        data_path: Path to the training data
        model_type: Type of model to use if training is needed
        force_retrain: If True, force retraining even if a model exists
        **train_kwargs: Additional arguments to pass to run_training_pipeline
        
    Returns:
        Tuple of (model, vectorizer)
    """
    # Try to load existing model if not forcing retrain
    if not force_retrain and model_exists():
        print("Loading existing model from disk...")
        model, vectorizer, _ = load_model()
        if model is not None and vectorizer is not None:
            return model, vectorizer
    
    # Train a new model if loading failed or forced
    print("Training new model...")
    results = run_training_pipeline(
        data_path=data_path,
        model_type=model_type,
        **train_kwargs
    )
    
    return results["model"], results["vectorizer"]


def run_training_pipeline(
    data_path: Path,
    model_type: str = DEFAULT_MODEL_TYPE,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
    save_to_disk: bool = True,
    cv_folds: int = 5,
    balance_classes: bool = True,
) -> dict:
    """
    Enhanced training pipeline with cross-validation and class balancing.
    
    Args:
        data_path: Path to the resume dataset directory
        model_type: Type of model to train ('naive_bayes' or 'logistic_regression')
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        max_features: Maximum number of features for TF-IDF vectorizer
        save_to_disk: Whether to save the trained model to disk
        cv_folds: Number of cross-validation folds
        balance_classes: Whether to balance class weights
        
    Returns:
        Dictionary containing model, vectorizer, and evaluation metrics
    """
    # Import required libraries
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.pipeline import Pipeline
    
    try:
        # Load and preprocess data
        print("Loading dataset...")
        df = load_resume_dataset(data_path)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df['category'])
        
        # Class distribution analysis
        class_counts = pd.Series(y_encoded).value_counts()
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            print(f"  {label_encoder.classes_[cls]}: {count} samples")
        
        # Split data with stratification
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['resume_text'],  # We'll preprocess in the pipeline
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Create preprocessing and modeling pipeline
        print("\nSetting up preprocessing and modeling pipeline...")
        
        # Define preprocessing steps
        preprocessor = Pipeline([
            ('tfidf', build_vectorizer(max_features=max_features)),
        ])
        
        # Add SMOTE to the pipeline if needed
        if balance_classes:
            print("Using SMOTE for class balancing...")
            model_pipeline = ImbPipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=random_state)),
                ('classifier', None)  # Will be set based on model_type
            ])
        else:
            model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', None)  # Will be set based on model_type
            ])
        
        # Define hyperparameter grids
        if model_type == "naive_bayes":
            param_grid = {
                'classifier__alpha': [0.1, 0.5, 1.0, 1.5],
                'classifier__fit_prior': [True, False],
                'preprocessor__tfidf__ngram_range': [(1, 1), (1, 2)],
                'preprocessor__tfidf__max_df': [0.8, 0.9, 1.0]
            }
            model_pipeline.set_params(classifier=MultinomialNB())
            
        else:  # logistic regression
            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['lbfgs', 'liblinear'],
                'classifier__class_weight': [None, 'balanced'],
                'preprocessor__tfidf__ngram_range': [(1, 1), (1, 2)],
                'preprocessor__tfidf__max_df': [0.8, 0.9, 1.0]
            }
            model_pipeline.set_params(classifier=LogisticRegression(
                max_iter=1000, 
                random_state=random_state
            ))
        
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=random_state
        )
        
        # Train model with cross-validation
        print(f"\nTraining {model_type} model with {cv_folds}-fold cross-validation...")
        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        y_pred = best_model.predict(X_test)
        
        # Get evaluation metrics
        metrics = evaluate_model(
            best_model, 
            X_test, 
            y_test,
            label_encoder=label_encoder
        )
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=label_encoder.classes_
        ))
        
        # Prepare results
        results = {
            'model': best_model,
            'vectorizer': best_model.named_steps['preprocessor'].named_steps['tfidf'],
            'metrics': metrics,
            'best_params': grid_search.best_params_,
            'label_encoder': label_encoder,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'class_distribution': dict(zip(label_encoder.classes_, np.bincount(y_encoded)))
        }
        
        # Save model if requested
        if save_to_disk:
            print("\nSaving model to disk...")
            save_model(
                model=best_model.named_steps['classifier'],
                vectorizer=best_model.named_steps['preprocessor'].named_steps['tfidf'],
                metrics=metrics,
                model_type=model_type,
                data_path=str(data_path)
            )
        
        return results
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train a resume classification model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("DATA/data"),
        help="Path to CSV dataset or directory of category subfolders",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["naive_bayes", "logistic_regression"],
        default="naive_bayes",
        help="Classifier to train",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of dataset to use for testing")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum number of TF-IDF features")
    parser.add_argument("--predict-text", type=str, default=None, help="Optional resume text to classify after training")

    args = parser.parse_args()

    results = run_training_pipeline(
        data_path=args.data_path,
        model_type=args.model_type,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )

    metrics = results["metrics"]
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Classification report:")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    if args.predict_text:
        predicted_category = predict_resume_category(results["model"], results["vectorizer"], args.predict_text)
        print("\nPredicted category for provided resume text:")
        print(predicted_category)


if __name__ == "__main__":
    main()
