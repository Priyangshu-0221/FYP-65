"""
Advanced Recommendation Engine for Internship Matching using Semantic Search

This module provides functionality to match student skills with internship opportunities
using state-of-the-art semantic search techniques (Sentence-Transformers).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Path to the dummy internship data
INTERNSHIP_DATA_PATH = Path("data/dummy_internship_recommendations.json")
# Define the 3 models for testing (Accuracy, Precision, Recall comparison)
AVAILABLE_MODELS = {
    "minilm": "all-MiniLM-L6-v2",           # Model 1: Your original fast baseline
    "bge-small": "BAAI/bge-small-en-v1.5",  # Model 2: State-of-the-Art research model (High precision)
    "mpnet": "all-mpnet-base-v2"            # Model 3: Most accurate heavy SBERT model
}

# Change this key to easily switch active models, or pass it via __init__
ACTIVE_MODEL_KEY = "minilm" 
DEFAULT_MODEL_NAME = AVAILABLE_MODELS[ACTIVE_MODEL_KEY]

class InternshipRecommendationEngine:
    """A recommendation engine for matching student skills with internship opportunities using AI embeddings."""
    
    def __init__(self, data_path: Optional[Path] = None, model_name: Optional[str] = None):
        """Initialize the recommendation engine with internship data.
        
        Args:
            data_path: Path to the JSON file containing internship data.
                       If None, uses the default path.
            model_name: Optional override for the semantic model. Can be used for testing.
        """
        self.data_path = data_path or INTERNSHIP_DATA_PATH
        self.internships = self._load_internship_data()
        self.skill_index = self._build_skill_index()
        
        # Use provided model or default to the active one
        self.active_model_name = model_name or DEFAULT_MODEL_NAME
        
        print(f"Loading semantic search model: {self.active_model_name}...")
        try:
            # Check for GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.active_model_name, device=device)
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Error loading model: {e}. Falling back to CPU.")
            self.model = SentenceTransformer(self.active_model_name, device="cpu")
            
        self.embeddings = self._build_semantic_embeddings()
    
    def _load_internship_data(self) -> List[Dict[str, Any]]:
        """Load internship data from the JSON file."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Internship data file not found at {self.data_path}. "
                "Please run create_dummy_recommendations.py first."
            )
        
        print(f"Loading internship data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, dict) and 'internships' in data:
                return data['internships']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"Invalid data format in {self.data_path}")
    
    def _build_skill_index(self) -> Dict[str, Set[int]]:
        """Build an index of skills to internship IDs for faster lookup."""
        index = defaultdict(set)
        
        for idx, internship in enumerate(self.internships):
            if not isinstance(internship, dict):
                continue
                
            skills_required = internship.get('skills_required', [])
            for skill in skills_required:
                if isinstance(skill, dict):
                    skill_name = skill.get('name', '').lower()
                    if skill_name:
                        index[skill_name].add(idx)
                elif isinstance(skill, str):
                    index[skill.lower()].add(idx)
                    
        return index
    
    def _build_semantic_embeddings(self) -> np.ndarray:
        """Compute embeddings for all internship descriptions."""
        print("Computing embeddings for internships...")
        documents = []
        valid_indices = []

        for idx, internship in enumerate(self.internships):
            if not isinstance(internship, dict):
                continue
            
            # Combine relevant text fields for a rich semantic representation
            title = internship.get('title', '')
            company = internship.get('company', '')
            description = internship.get('description', '')
            requirements = ' '.join(str(r) for r in internship.get('requirements', []))
            
            # Helper to get skill string safely
            skills = []
            for s in internship.get('skills_required', []):
                if isinstance(s, dict):
                    skills.append(s.get('name', ''))
                elif isinstance(s, str):
                    skills.append(s)
            skills_str = ', '.join(skills)

            # Creating a comprehensive text representation
            text = f"{title} at {company}. {description} Requirements: {requirements} Skills: {skills_str}"
            
            if text.strip():
                documents.append(text)
                valid_indices.append(idx)

        if not documents:
            return np.zeros((len(self.internships), 0))

        try:
            # Batch encode for efficiency
            embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
            
            # Map back to full array (handle potentially skipped items)
            full_embeddings = np.zeros((len(self.internships), embeddings.shape[1]))
            full_embeddings[valid_indices] = embeddings
            
            print(f"Computed embeddings with shape {full_embeddings.shape}")
            return full_embeddings
            
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            return np.zeros((len(self.internships), 0))

    def _calculate_semantic_similarity(self, query_text: str) -> np.ndarray:
        """Calculate semantic similarity between query and stored internships."""
        if not query_text or self.embeddings.shape[1] == 0:
            return np.zeros(len(self.internships))
            
        try:
            # Encode user query
            query_embedding = self.model.encode(query_text, convert_to_numpy=True)
            
            # Calculate cosine similarity
            # Reshape query to (1, embedding_dim)
            similarities = cosine_similarity([query_embedding], self.embeddings)
            return similarities.flatten()
            
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return np.zeros(len(self.internships))
    
    def _calculate_marks_similarity(self, user_marks: Dict[str, float]) -> np.ndarray:
        """Calculate scores based on academic marks."""
        scores = np.ones(len(self.internships)) * 0.5  # Default neutral score
        
        # Pre-compile regex for performance
        cgpa_regex = re.compile(r'cumm?ulative gpa.*?([0-9]\.[0-9]|[0-9]\s*[+\-]?\s*[0-9]*)', re.IGNORECASE)
        percent_regex = re.compile(r'(?:minimum|min\.?|at least)?\s*([0-9]+)%', re.IGNORECASE)

        for idx, internship in enumerate(self.internships):
            description = (internship.get('description', '') + ' ' + 
                         ' '.join(internship.get('requirements', []))).lower()
            
            # Check CGPA
            cgpa_match = cgpa_regex.search(description)
            if cgpa_match:
                try:
                    required = float(cgpa_match.group(1))
                    user_cgpa = user_marks.get('cgpa', 0)
                    scores[idx] = 1.0 if user_cgpa >= required else max(0.1, user_cgpa / required)
                except (ValueError, TypeError):
                    pass
            
            # Check Percentage
            percent_match = percent_regex.search(description)
            if percent_match:
                try:
                    required = float(percent_match.group(1))
                    user_percent = user_marks.get('percentage', 0)
                    if user_percent >= required:
                        scores[idx] = max(scores[idx], 1.0)
                    else:
                        scores[idx] = max(scores[idx], max(0.1, user_percent / required))
                except (ValueError, TypeError):
                    pass
                    
        return scores

    def recommend_internships(
        self,
        user_skills: List[str],
        user_marks: Optional[Dict[str, float]] = None,
        skill_count: Optional[int] = None,
        top_n: int = 10,
        semantic_weight: float = 0.7,  # Primary driver
        keyword_weight: float = 0.2,   # Secondary check
        marks_weight: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered internship recommendations."""
        if not user_skills or not self.internships:
            return []
            
        try:
            num_internships = len(self.internships)
            
            # 1. Semantic Search (The Core Brain)
            # Create a rich query representation from user profile
            query_skills = ", ".join(user_skills)
            query_text = f"Looking for internship requiring skills: {query_skills}."
            
            semantic_scores = self._calculate_semantic_similarity(query_text)
            
            # 2. Keyword Exact Match (The Verify)
            # Boosts internships that explicitly list the skill
            keyword_scores = np.zeros(num_internships)
            user_skills_set = set(s.lower() for s in user_skills)
            
            for idx, internship in enumerate(self.internships):
                # Count matches
                int_skills = internship.get('skills_required', [])
                matches = 0
                for s in int_skills:
                    s_name = s.get('name', '') if isinstance(s, dict) else s
                    if s_name.lower() in user_skills_set:
                        matches += 1
                
                # Normalize by total required skills to prevent bias towards long lists
                total_req = max(1, len(int_skills))
                keyword_scores[idx] = matches / total_req
            
            # 3. Marks Check (The Gatekeeper)
            marks_scores = np.ones(num_internships) * 0.5
            if user_marks:
                marks_scores = self._calculate_marks_similarity(user_marks)
            
            # 4. Combine Scores
            total_scores = (
                semantic_weight * semantic_scores +
                keyword_weight * keyword_scores +
                marks_weight * marks_scores
            )
            
            # Get Top N
            top_indices = np.argsort(total_scores)[::-1][:top_n]
            
            recommendations = []
            for idx in top_indices:
                if idx < num_internships and total_scores[idx] > 0.15: # Threshold to filter noise
                    internship = self.internships[idx].copy()
                    
                    # Add explanation metadata 
                    internship['score'] = float(total_scores[idx])
                    internship['relevance_metrics'] = {
                        'semantic_match': f"{semantic_scores[idx]:.2f}",
                        'keyword_match': f"{keyword_scores[idx]:.2f}"
                    }
                    recommendations.append(internship)
            
            return recommendations

        except Exception as e:
            print(f"Error in recommendation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return []

def load_recommendation_engine() -> InternshipRecommendationEngine:
    return InternshipRecommendationEngine()
