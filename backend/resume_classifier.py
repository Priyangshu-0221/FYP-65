import argparse
import re
from pathlib import Path
from typing import Union, Optional, Tuple

import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from skill_extractor import ensure_nltk_dependencies


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
    lowered = text.lower()
    letters_only = re.sub(r"[^a-z\s]", " ", lowered)
    normalized = re.sub(r"\s+", " ", letters_only)
    return normalized.strip()


def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    processed_tokens = []
    for token in tokens:
        if token in stop_words or len(token) <= 2:
            continue
        lemma = lemmatizer.lemmatize(token)
        processed_tokens.append(lemma)
    return " ".join(processed_tokens)


def preprocess_corpus(text_series: pd.Series) -> list[str]:
    ensure_nltk_dependencies()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return [preprocess_text(text, stop_words, lemmatizer) for text in text_series]


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


def vectorize_text(train_texts: list[str], test_texts: list[str], vectorizer: TfidfVectorizer | None = None):
    if vectorizer is None:
        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(train_texts)
    else:
        X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return vectorizer, X_train, X_test


def train_model(X_train, y_train, model_type: str):
    if model_type == "naive_bayes":
        model = MultinomialNB()
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("model_type must be 'naive_bayes' or 'logistic_regression'")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    return {"accuracy": accuracy, "classification_report": report, "confusion_matrix": matrix, "y_pred": y_pred}


def predict_resume_category(model, vectorizer, resume_text: str) -> str:
    processed_text = preprocess_corpus(pd.Series([resume_text]))[0]
    features = vectorizer.transform([processed_text])
    return model.predict(features)[0]


def run_training_pipeline(
    data_path: Path,
    model_type: str = "naive_bayes",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
) -> dict:
    df = load_resume_dataset(data_path)
    processed_texts = preprocess_corpus(df["Resume"])

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        processed_texts, df["Category"], test_size=test_size, random_state=random_state, stratify=df["Category"]
    )

    vectorizer = build_vectorizer(max_features=max_features)
    vectorizer, X_train, X_test = vectorize_text(X_train_texts, X_test_texts, vectorizer)

    model = train_model(X_train, y_train, model_type=model_type)
    metrics = evaluate_model(model, X_test, y_test)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


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
