"""
Process all resumes in DATA folder and save to Excel files
Creates 3 Excel files:
1. raw_data.xlsx - Original extracted text
2. cleaned_data.xlsx - Cleaned text (lowercase, no special chars)
3. tokenized_data.xlsx - Tokenized words
"""

import re
import sys
import os
from pathlib import Path

# Add project root to Python path so we can import backend modules
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from backend.extractor import process_resume
from tqdm import tqdm


def clean_text(text: str) -> str:
    """Clean text: lowercase, remove special characters"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize_text(text: str) -> str:
    """Tokenize text into words"""
    if not text:
        return ""
    
    # Split into words
    words = text.split()
    
    # Remove very short words (less than 2 characters)
    words = [w for w in words if len(w) >= 2]
    
    return ' '.join(words)


def process_all_resumes(data_dir: str, max_resumes: int = None):
    """
    Process all resumes in the DATA folder
    
    Args:
        data_dir: Path to DATA/data folder containing category folders
        max_resumes: Maximum number of resumes to process (None = all)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: {data_dir} not found")
        return
    
    # Lists to store data for each Excel file
    raw_data = []
    cleaned_data = []
    tokenized_data = []
    
    # Get all category folders
    categories = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(categories)} categories")
    
    total_processed = 0
    
    # Process each category
    for category_dir in categories:
        category_name = category_dir.name
        print(f"\nProcessing category: {category_name}")
        
        # Get all PDFs in this category
        pdf_files = list(category_dir.glob("*.pdf"))
        
        # Limit if max_resumes specified
        if max_resumes and total_processed >= max_resumes:
            break
        
        if max_resumes:
            remaining = max_resumes - total_processed
            pdf_files = pdf_files[:remaining]
        
        # Process each PDF
        for pdf_file in tqdm(pdf_files, desc=f"{category_name}"):
            result = process_resume(str(pdf_file), category_name)
            
            if result:
                # Raw data
                raw_data.append({
                    'filename': result['filename'],
                    'category': result['category'],
                    'text': result['text'],
                    'skills': ', '.join(result['skills']),
                    'skills_count': result['skills_count'],
                    'education': result['education'],
                    'experience': result['experience'],
                    'cgpa': result['cgpa'],
                    'percentage': result['percentage']
                })
                
                # Cleaned data
                cleaned_text = clean_text(result['text'])
                cleaned_data.append({
                    'filename': result['filename'],
                    'category': result['category'],
                    'cleaned_text': cleaned_text,
                    'skills': ', '.join(result['skills']),
                    'skills_count': result['skills_count']
                })
                
                # Tokenized data
                tokenized_text = tokenize_text(cleaned_text)
                tokenized_data.append({
                    'filename': result['filename'],
                    'category': result['category'],
                    'tokenized_text': tokenized_text,
                    'token_count': len(tokenized_text.split()),
                    'skills': ', '.join(result['skills']),
                    'skill_count': result['skills_count']
                })
                
                total_processed += 1
    
    print(f"\n\nTotal resumes processed: {total_processed}")
    
    # Save to Excel files
    # Save to Excel files
    print("\nSaving to Excel files...")
    
    # Save processed data in DATA/processed, relative to the project root
    output_dir = Path(data_dir).parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    df_raw = pd.DataFrame(raw_data)
    raw_file = output_dir / "raw_data.xlsx"
    df_raw.to_excel(raw_file, index=False)
    print(f"[OK] Saved: {raw_file} ({len(df_raw)} rows)")
    
    # Save cleaned data
    df_cleaned = pd.DataFrame(cleaned_data)
    cleaned_file = output_dir / "cleaned_data.xlsx"
    df_cleaned.to_excel(cleaned_file, index=False)
    print(f"[OK] Saved: {cleaned_file} ({len(df_cleaned)} rows)")
    
    # Save tokenized data
    df_tokenized = pd.DataFrame(tokenized_data)
    tokenized_file = output_dir / "tokenized_data.xlsx"
    df_tokenized.to_excel(tokenized_file, index=False)
    print(f"[OK] Saved: {tokenized_file} ({len(df_tokenized)} rows)")
    
    print("\n=== ALL DONE ===")
    return df_raw, df_cleaned, df_tokenized


if __name__ == "__main__":
    # Process first 50 resumes as a test
    # Change to None to process ALL resumes
    
    # Use Path to handle the directory correctly
    script_dir = Path(__file__).parent
    # Project root is one level up from scripts
    project_root = script_dir.parent
    data_directory = project_root / "DATA" / "data"
    
    print("=== Resume Processing Pipeline ===\n")
    print(f"Looking for resumes in: {data_directory}\n")
    print("This will create 3 Excel files:")
    print("1. raw_data.xlsx - Original text + extracted info")
    print("2. cleaned_data.xlsx - Cleaned text")
    print("3. tokenized_data.xlsx - Tokenized text")
    print("\nProcessing ALL resumes in the dataset...\n")
    
    process_all_resumes(str(data_directory), max_resumes=None)
