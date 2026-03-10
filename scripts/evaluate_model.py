import os
import sys
import time
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


# Add backend to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backend.recommendation_engine import InternshipRecommendationEngine

# 1. Map Resume Categories to Internship Categories
CATEGORY_MAPPING = {
    'ACCOUNTANT': ['Finance', 'Accounting'],
    'ADVOCATE': ['Legal'],
    'AGRICULTURE': ['Agriculture'],
    'APPAREL': ['Fashion'],
    'ARTS': ['Design', 'Art'],
    'AUTOMOBILE': ['Engineering', 'Manufacturing'],
    'AVIATION': ['Logistics', 'Engineering'],
    'BANKING': ['Finance'],
    'BPO': ['Support', 'Sales', 'Customer Service'],
    'BUSINESS-DEVELOPMENT': ['Sales', 'Business'],
    'CHEF': ['Hospitality'],
    'CONSTRUCTION': ['Engineering', 'Civil'],
    'CONSULTANT': ['Sales', 'Consulting'],
    'DESIGNER': ['Design', 'UX/UI', 'Graphic Design'],
    'DIGITAL-MEDIA': ['Marketing', 'PR', 'Media'],
    'ENGINEERING': ['Engineering', 'IT', 'Software', 'Mechanical', 'Electrical', 'Civil'],
    'FINANCE': ['Finance', 'Accounting'],
    'FITNESS': ['Healthcare', 'Sports'],
    'HEALTHCARE': ['Healthcare', 'Medicine', 'Science'],
    'HR': ['HR', 'Human Resources'],
    'INFORMATION-TECHNOLOGY': [
        'Software', 'Web', 'Data', 'AI', 'Cloud', 'Security', 'Mobile', 
        'IT', 'Information Technology'
    ],
    'PUBLIC-RELATIONS': ['PR', 'Marketing', 'Communications'],
    'SALES': ['Sales', 'Business Development'],
    'TEACHER': ['Education', 'Teaching', 'Training']
}

def load_data(limit=100):
    """Load cleaned dataset of resumes for evaluation"""
    data_path = project_root / 'DATA' / 'processed' / 'cleaned_data.xlsx'
    if not data_path.exists():
        print(f"Dataset not found at: {data_path}. Generating dummy evaluation data instead...")
        # Try raw_data
        alt_path = project_root / 'DATA' / 'processed' / 'raw_data.xlsx'
        if alt_path.exists():
            df = pd.read_excel(alt_path)
            if limit is None:
                return df
            return df.sample(n=min(limit, len(df)), random_state=42)
        return pd.DataFrame() # Fallback empty
    
    df = pd.read_excel(data_path)
    if limit is None:
        return df
    return df.sample(n=min(limit, len(df)), random_state=42)

def run_evaluation():
    print("="*60)
    print("Model Evaluation: all-MiniLM-L6-v2")
    print("Metrics: Precision@k, Recall@k, F1@k, Latency, Robustness")
    print("="*60)
    
    # 1. Initialize Engine
    print("\nLoading Recommendation Engine & Embedding Model...")
    # Overriding internship data path as it's run from scripts
    data_path = project_root / 'backend' / 'data' / 'dummy_internship_recommendations.json'
    engine = InternshipRecommendationEngine(data_path=data_path)
    
    # Pre-calculate counts of internships per category/domain for Recall
    all_internships = engine.internships
    internship_cat_counts = {}
    for intent in all_internships:
        cat = intent.get('domain', intent.get('category', 'Unknown'))
        internship_cat_counts[cat] = internship_cat_counts.get(cat, 0) + 1

    # 2. Load Resume Data
    print("\nLoading dataset of resumes for evaluation...")
    df = load_data(limit=None) # Process ALL resumes
    if df.empty:
        print("Dataset is empty. Cannot perform evaluation.")
        return

    print(f"Evaluating on {len(df)} resumes.")

    # Tracking Metrics
    latencies = []
    precision_k = []
    recall_k = []
    accuracy_scores = []
    
    # Robustness noise test counter
    noise_drops = []

    k = 5 # Evaluate Top-5 recommendations

    print("\nProcessing & Calculating Metrics (Semantic Similarity)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        category = row.get('category', '').upper()
        if category not in CATEGORY_MAPPING:
            continue
            
        target_internship_categories = CATEGORY_MAPPING[category]
        skills_str = str(row.get('skills', ''))
        skills = [s.strip() for s in skills_str.split(',') if s.strip()]
        
        # Calculate total relevant documentaries in database for this category 
        total_relevant = 0
        for cat in internship_cat_counts.keys():
            if any(target.lower() in str(cat).lower() for target in target_internship_categories):
                total_relevant += internship_cat_counts[cat]
                
        if total_relevant == 0:
            continue # If database has no matches for this category, skip
        
        # --- Normal Inference Latency & Accuracy ---
        start_time = time.perf_counter()
        recs = engine.recommend_internships(
            user_skills=skills, 
            top_n=k,
            semantic_weight=0.8, # Rely heavily on MiniLM
            keyword_weight=0.2,
            marks_weight=0.0
        )
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # In ms
        
        # Calculate Hits
        hits = 0
        for r in recs:
            r_cat = r.get('domain', r.get('category', ''))
            if any(target.lower() in str(r_cat).lower() for target in target_internship_categories):
                hits += 1
                
        # Precision@k = Hits / k
        prec = hits / max(1, len(recs)) if recs else 0
        precision_k.append(prec)
        
        # Accuracy@k = 1 if at least one hit, else 0
        accuracy_scores.append(1 if hits > 0 else 0)
        
        # Recall@k = Hits / total_relevant_in_database
        rec = hits / total_relevant
        recall_k.append(rec)
        
        # --- Robustness Test (Missing/Noisy Data) ---
        # Randomly drop 50% of skills to see if it still recommends within boundary
        if len(skills) > 2:
            noisy_skills = random.sample(skills, max(1, len(skills) // 2))
            recs_noisy = engine.recommend_internships(user_skills=noisy_skills, top_n=k)
            
            hits_noisy = 0
            for r in recs_noisy:
                r_cat = r.get('domain', r.get('category', ''))
                if any(target.lower() in str(r_cat).lower() for target in target_internship_categories):
                    hits_noisy += 1
            
            # Did the accuracy drop substantially? (< 30% drop is robust)
            drop_percentage = (hits - hits_noisy) / max(1, hits)
            noise_drops.append(max(0, drop_percentage))


    # Compile stats
    avg_precision = np.mean(precision_k) if precision_k else 0
    avg_recall = np.mean(recall_k) if recall_k else 0
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
    
    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    
    avg_noise_drop = np.mean(noise_drops) if noise_drops else 0
    robustness_score = 100 - (avg_noise_drop * 100)

    # 3. Print the Final Report
    print("\n" + "="*60)
    print("EVALUATION RESULTS: all-MiniLM-L6-v2")
    print("="*60)
    
    data = [
        ["Accuracy@5", f"{avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)"],
        ["Precision@5", f"{avg_precision:.4f} ({avg_precision*100:.1f}%)"],
        ["Recall@5", f"{avg_recall:.4f} ({avg_recall*100:.1f}%)"],
        ["F1-Score", f"{f1_score:.4f} ({f1_score*100:.1f}%)"],
        ["Avg Inference Latency", f"{avg_latency:.2f} ms / request"],
        ["95th Percentile Latency", f"{p95_latency:.2f} ms / request"],
        ["Robustness (50% Data Loss)", f"{robustness_score:.1f}% Retention Rating"],
    ]
    
    for row in data:
        padded_name = str(row[0]).ljust(35)
        padded_val = str(row[1])
        print(f"| {padded_name} | {padded_val}")
    print("+" + "-"*37 + "+" + "-"*50 + "+")
    
    print("\nInsights:")
    print("1. Precision measures if the TOP 5 recommended jobs matched the candidate's core domain.")
    print("2. Recall measures the model's ability to find ALL relevant jobs in the static database.")
    print("3. Latency is the real-world computation time of Semantic Similarities between the User Profile embedding and Internship Embeddings.")
    print("4. The system proved {:.1f}% robust when 50% of the candidate's skills were corrupted or missing.".format(robustness_score))
    print("="*60)

    # 4. Save results to Excel
    metrics_path = project_root / 'DATA' / 'processed' / 'evaluation_metrics.xlsx'
    
    # Create DataFrame from data, including exact float values for tracking
    export_data = {
        "Metric": [row[0] for row in data],
        "Value_String": [row[1] for row in data],
        "Raw_Score": [avg_accuracy, avg_precision, avg_recall, f1_score, avg_latency, p95_latency, robustness_score]
    }
    
    df_metrics = pd.DataFrame(export_data)
    df_metrics.to_excel(metrics_path, index=False)
    print(f"\n[OK] Evaluation metrics successfully saved to: {metrics_path}")

if __name__ == "__main__":
    run_evaluation()
