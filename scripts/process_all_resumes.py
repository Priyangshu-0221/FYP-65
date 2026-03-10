# """
# Process all resumes in DATA folder and save to Excel files
# Creates 3 Excel files:
# 1. raw_data.xlsx - Original extracted text
# 2. cleaned_data.xlsx - Cleaned text (lowercase, no special chars)
# 3. tokenized_data.xlsx - Tokenized words
# """

# import re
# import sys
# import os
# from pathlib import Path

# # Add project root to Python path so we can import backend modules
# sys.path.append(str(Path(__file__).parent.parent))

# import pandas as pd
# from backend.extractor import process_resume
# from tqdm import tqdm


# def clean_text(text: str) -> str:
#     """Clean text: lowercase, remove special characters"""
#     if not text:
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove special characters, keep only letters and spaces
#     text = re.sub(r'[^a-z\s]', ' ', text)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
    
#     return text.strip()


# def tokenize_text(text: str) -> str:
#     """Tokenize text into words"""
#     if not text:
#         return ""
    
#     # Split into words
#     words = text.split()
    
#     # Remove very short words (less than 2 characters)
#     words = [w for w in words if len(w) >= 2]
    
#     return ' '.join(words)


# def process_all_resumes(data_dir: str, max_resumes: int = None):
#     """
#     Process all resumes in the DATA folder
    
#     Args:
#         data_dir: Path to DATA/data folder containing category folders
#         max_resumes: Maximum number of resumes to process (None = all)
#     """
#     data_path = Path(data_dir)
    
#     if not data_path.exists():
#         print(f"Error: {data_dir} not found")
#         return
    
#     # Lists to store data for each Excel file
#     raw_data = []
#     cleaned_data = []
#     tokenized_data = []
    
#     # Get all category folders
#     categories = [d for d in data_path.iterdir() if d.is_dir()]
    
#     print(f"Found {len(categories)} categories")
    
#     total_processed = 0
    
#     # Process each category
#     for category_dir in categories:
#         category_name = category_dir.name
#         print(f"\nProcessing category: {category_name}")
        
#         # Get all PDFs in this category
#         pdf_files = list(category_dir.glob("*.pdf"))
        
#         # Limit if max_resumes specified
#         if max_resumes and total_processed >= max_resumes:
#             break
        
#         if max_resumes:
#             remaining = max_resumes - total_processed
#             pdf_files = pdf_files[:remaining]
        
#         # Process each PDF
#         for pdf_file in tqdm(pdf_files, desc=f"{category_name}"):
#             result = process_resume(str(pdf_file), category_name)
            
#             if result:
#                 # Raw data
#                 raw_data.append({
#                     'filename': result['filename'],
#                     'category': result['category'],
#                     'text': result['text'],
#                     'skills': ', '.join(result['skills']),
#                     'skills_count': result['skills_count'],
#                     'education': result['education'],
#                     'experience': result['experience'],
#                     'cgpa': result['cgpa'],
#                     'percentage': result['percentage']
#                 })
                
#                 # Cleaned data
#                 cleaned_text = clean_text(result['text'])
#                 cleaned_data.append({
#                     'filename': result['filename'],
#                     'category': result['category'],
#                     'cleaned_text': cleaned_text,
#                     'skills': ', '.join(result['skills']),
#                     'skills_count': result['skills_count']
#                 })
                
#                 # Tokenized data
#                 tokenized_text = tokenize_text(cleaned_text)
#                 tokenized_data.append({
#                     'filename': result['filename'],
#                     'category': result['category'],
#                     'tokenized_text': tokenized_text,
#                     'token_count': len(tokenized_text.split()),
#                     'skills': ', '.join(result['skills']),
#                     'skill_count': result['skills_count']
#                 })
                
#                 total_processed += 1
    
#     print(f"\n\nTotal resumes processed: {total_processed}")
    
#     # Save to Excel files
#     # Save to Excel files
#     print("\nSaving to Excel files...")
    
#     # Save processed data in DATA/processed, relative to the project root
#     output_dir = Path(data_dir).parent / "processed"
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Save raw data
#     df_raw = pd.DataFrame(raw_data)
#     raw_file = output_dir / "raw_data.xlsx"
#     df_raw.to_excel(raw_file, index=False)
#     print(f"[OK] Saved: {raw_file} ({len(df_raw)} rows)")
    
#     # Save cleaned data
#     df_cleaned = pd.DataFrame(cleaned_data)
#     cleaned_file = output_dir / "cleaned_data.xlsx"
#     df_cleaned.to_excel(cleaned_file, index=False)
#     print(f"[OK] Saved: {cleaned_file} ({len(df_cleaned)} rows)")
    
#     # Save tokenized data
#     df_tokenized = pd.DataFrame(tokenized_data)
#     tokenized_file = output_dir / "tokenized_data.xlsx"
#     df_tokenized.to_excel(tokenized_file, index=False)
#     print(f"[OK] Saved: {tokenized_file} ({len(df_tokenized)} rows)")
    
#     print("\n=== ALL DONE ===")
#     return df_raw, df_cleaned, df_tokenized


# if __name__ == "__main__":
#     # Process first 50 resumes as a test
#     # Change to None to process ALL resumes
    
#     # Use Path to handle the directory correctly
#     script_dir = Path(__file__).parent
#     # Project root is one level up from scripts
#     project_root = script_dir.parent
#     data_directory = project_root / "DATA" / "data"
    
#     print("=== Resume Processing Pipeline ===\n")
#     print(f"Looking for resumes in: {data_directory}\n")
#     print("This will create 3 Excel files:")
#     print("1. raw_data.xlsx - Original text + extracted info")
#     print("2. cleaned_data.xlsx - Cleaned text")
#     print("3. tokenized_data.xlsx - Tokenized text")
#     print("\nProcessing ALL resumes in the dataset...\n")
    
#     process_all_resumes(str(data_directory), max_resumes=None)
"""
Resume Feature Extraction Pipeline: Clean text, tokenize, and generate TF-IDF/BERT features.
"""

import re
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizer, BertModel

# Add project root to path for local imports
try:
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    from backend.extractor import process_resume
except ImportError:
    pass  # Allow script to load even if backend imports fail

# --- Constants ---
RAW_DATA_FILE = "raw_data.xlsx"
CLEANED_DATA_FILE = "cleaned_data.xlsx"
TOKENIZED_DATA_FILE = "tokenized_data.xlsx"
TFIDF_FEATURES_FILE = "tfidf_features.xlsx"
BERT_EMBEDDINGS_FILE = "bert_embeddings.xlsx"

BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 512
MIN_WORD_LENGTH = 2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# --- Text Processing ---

def clean_text(text: Optional[str]) -> str:
    """Lowercase text, remove non-alphabetic chars, and normalize whitespace."""
    if not text: return ""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', ' ', text.lower())).strip()


def tokenize_text(text: Optional[str]) -> str:
    """Filter out short words and return space-separated tokens."""
    if not text: return ""
    return ' '.join([w for w in text.split() if len(w) >= MIN_WORD_LENGTH])


# --- Feature Extraction ---

def generate_tfidf_features(texts: pd.Series) -> Tuple[Any, TfidfVectorizer]:
    """Generate TF-IDF features (max 3000 features, unigrams/bigrams)."""
    logger.info("Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
    return vectorizer.fit_transform(texts), vectorizer


def generate_bert_embeddings(texts: pd.Series, device: torch.device) -> np.ndarray:
    """Generate BERT CLS token embeddings for input texts."""
    logger.info(f"Loading BERT model ({BERT_MODEL_NAME})...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(device).eval()

    embeddings = []
    logger.info("Generating BERT embeddings...")
    
    with torch.no_grad():
        for text in tqdm(texts, desc="BERT Processing"):
            inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors='pt').to(device)
            embeddings.append(model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().flatten())

    return np.array(embeddings)


# --- Main Pipeline ---

def process_resumes_pipeline(data_dir: Path) -> None:
    """Run extraction pipeline: load PDFs, clean text, tokenize, and save features."""
    output_dir = data_dir.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        logger.error(f"Data directory missing: {data_dir}")
        return

    raw_data, cleaned_data, tokenized_data = [], [], []
    categories = [d for d in data_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(categories)} categories.")

    for category_dir in categories:
        for pdf_file in tqdm(list(category_dir.glob("*.pdf")), desc=f"Processing {category_dir.name}"):
            try:
                result = process_resume(str(pdf_file), category_dir.name)
                if result and result.get('text'):
                    cleaned = clean_text(result['text'])
                    base_info = {
                        'filename': result.get('filename', pdf_file.name), 
                        'category': category_dir.name,
                        'skills': ', '.join(result.get('skills', [])), # Join list to string
                        'education': result.get('education', ''),
                        'experience': result.get('experience', ''),
                        'cgpa': result.get('cgpa', ''),
                        'percentage': result.get('percentage', '')
                    }
                    
                    # Create focused text for models
                    focused_text = f"Skills: {base_info['skills']}. Experience: {base_info['experience']}. Education: {base_info['education']}. CGPA: {base_info['cgpa']}. Percentage: {base_info['percentage']}."
                    cleaned_focused = clean_text(focused_text)
                    
                    raw_data.append({**base_info, 'text': result['text']})
                    # Save focused text in cleaned/tokenized data instead of full text
                    cleaned_data.append({**base_info, 'cleaned_text': cleaned_focused})
                    tokenized_data.append({**base_info, 'tokenized_text': tokenize_text(cleaned_focused)})
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")

    if not raw_data:
        logger.warning("No data processed.")
        return

    # Save intermediate files
    logger.info("Saving processed Excel files...")
    pd.DataFrame(raw_data).to_excel(output_dir / RAW_DATA_FILE, index=False)
    df_cleaned = pd.DataFrame(cleaned_data)
    df_cleaned.to_excel(output_dir / CLEANED_DATA_FILE, index=False)
    pd.DataFrame(tokenized_data).to_excel(output_dir / TOKENIZED_DATA_FILE, index=False)

    # TF-IDF Features
    try:
        X_tfidf, vectorizer = generate_tfidf_features(df_cleaned['cleaned_text'])
        tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
        
        # Insert metadata columns
        metadata_cols = ['category', 'filename', 'skills', 'education', 'experience', 'cgpa', 'percentage']
        
        # Handle column collisions (if a word in text matches a metadata column name)
        for col in metadata_cols:
            if col in tfidf_df.columns:
                tfidf_df.rename(columns={col: f"{col}_token"}, inplace=True)
        
        for col in reversed(metadata_cols):
            if col in df_cleaned.columns:
                tfidf_df.insert(0, col, df_cleaned[col])
                
        tfidf_df.to_excel(output_dir / TFIDF_FEATURES_FILE, index=False)
        logger.info("Saved TF-IDF features.")
    except Exception as e:
        logger.error(f"TF-IDF failed: {e}")

    # BERT Embeddings
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_embeddings = generate_bert_embeddings(df_cleaned['cleaned_text'], device)
        bert_df = pd.DataFrame(bert_embeddings)
        
        # Insert metadata columns
        metadata_cols = ['category', 'filename', 'skills', 'education', 'experience', 'cgpa', 'percentage']
        for col in reversed(metadata_cols):
            if col in df_cleaned.columns:
                bert_df.insert(0, col, df_cleaned[col])
                
        bert_df.to_excel(output_dir / BERT_EMBEDDINGS_FILE, index=False)
        logger.info("Saved BERT embeddings.")
    except Exception as e:
        logger.error(f"BERT embeddings failed: {e}")

    logger.info("Pipeline Complete.")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "DATA" / "data"
    print(f"\n--- Resume Processing Pipeline ---\nTarget: {data_path}\n")
    process_resumes_pipeline(data_path)