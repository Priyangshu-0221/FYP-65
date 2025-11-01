"""
Advanced Recommendation Engine for Internship Matching

This module provides functionality to match student skills with internship opportunities
using a combination of content-based and collaborative filtering techniques.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Path to the dummy internship data
INTERNSHIP_DATA_PATH = Path("data/dummy_internship_recommendations.json")

class InternshipRecommendationEngine:
    """A recommendation engine for matching student skills with internship opportunities."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the recommendation engine with internship data.
        
        Args:
            data_path: Path to the JSON file containing internship data.
                       If None, uses the default path.
        """
        self.data_path = data_path or INTERNSHIP_DATA_PATH
        self.internships = self._load_internship_data()
        self.skill_index = self._build_skill_index()
        self.vectorizer, self.tfidf_matrix = self._build_tfidf_model()
    
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
            
            # Handle the case where data is a dict with 'internships' key
            if isinstance(data, dict) and 'internships' in data:
                print(f"Found {len(data['internships'])} internships in data")
                if data['internships'] and isinstance(data['internships'][0], dict):
                    print(f"First internship ID: {data['internships'][0].get('id', 'N/A')}")
                return data['internships']
            # Handle the case where data is directly a list
            elif isinstance(data, list):
                print(f"Found {len(data)} internships in data")
                if data and isinstance(data[0], dict):
                    print(f"First internship ID: {data[0].get('id', 'N/A')}")
                return data
            else:
                raise ValueError(f"Invalid data format. Expected a list of internships or a dict with 'internships' key. Got: {type(data)}")
    
    def _build_skill_index(self) -> Dict[str, Set[int]]:
        """Build an index of skills to internship IDs for faster lookup."""
        index = defaultdict(set)
        print(f"Building skill index for {len(self.internships)} internships")
        
        for idx, internship in enumerate(self.internships):
            if not isinstance(internship, dict):
                print(f"Warning: Internship at index {idx} is not a dictionary: {internship}")
                continue
                
            skills_required = internship.get('skills_required', [])
            if not isinstance(skills_required, (list, tuple)):
                print(f"Warning: skills_required for internship {idx} is not a list: {skills_required}")
                continue
                
            print(f"Processing internship {idx} with skills: {skills_required}")
                
            for skill in skills_required:
                if isinstance(skill, dict):
                    # Handle case where skills are objects with 'name' and 'proficiency'
                    skill_name = skill.get('name', '').lower()
                    if skill_name:
                        index[skill_name].add(idx)
                elif isinstance(skill, str):
                    # Handle case where skills are simple strings
                    index[skill.lower()].add(idx)
                    
        print(f"Built index with {len(index)} unique skills")
        return index
    
    def _build_tfidf_model(self) -> Tuple[TfidfVectorizer, np.ndarray]:
        """Build a TF-IDF model for internship descriptions and requirements."""
        # Combine description and requirements into a single text field
        documents = []
        valid_indices = []  # Keep track of which internships have valid documents
        print("Building TF-IDF model...")
        
        for idx, internship in enumerate(self.internships):
            if not isinstance(internship, dict):
                print(f"Skipping non-dictionary internship at index {idx}")
                continue
                
            # Get description
            description = internship.get('description', '')
            if not isinstance(description, str):
                description = str(description) if description is not None else ""
                
            # Get requirements
            requirements = internship.get('requirements', [])
            if not isinstance(requirements, (list, tuple)):
                requirements = []
                
            # Combine into a single text
            text = f"{description} {' '.join(str(r) for r in requirements)}"
            
            # Only add non-empty documents
            if text.strip():
                documents.append(text.lower())
                valid_indices.append(idx)
            
            # Debug output for first few internships
            if idx < 3:  # Print details for first 3 internships
                print(f"Document {idx} sample: {text[:100]}...")
        
        print(f"Processed {len(documents)} valid documents out of {len(self.internships)} internships")
        
        if not documents:
            # If no valid documents, return empty matrix with correct dimensions
            empty_matrix = np.zeros((len(self.internships), 0))
            return TfidfVectorizer(), empty_matrix
        
        try:
            # Create and fit the TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,  # Reduced min_df to ensure we don't filter out too much
                max_df=0.8
            )
            
            print("Fitting TF-IDF vectorizer...")
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            if len(documents) > 0:
                print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
                print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
            
            # Create a full-sized matrix with zeros for invalid documents
            full_tfidf = np.zeros((len(self.internships), tfidf_matrix.shape[1]))
            full_tfidf[valid_indices] = normalize(tfidf_matrix, axis=1)
            
            return vectorizer, full_tfidf
            
        except Exception as e:
            print(f"Error in TF-IDF processing: {str(e)}")
            print(f"Document lengths: [{' '.join(str(len(d)) for d in documents[:5])}...]")
            # Return empty matrix with correct dimensions in case of error
            empty_matrix = np.zeros((len(self.internships), 0))
            return TfidfVectorizer(), empty_matrix
    
    def _calculate_skill_similarity(self, user_skills: List[str]) -> np.ndarray:
        """Calculate similarity scores based on skill overlap with weighting."""
        if not user_skills:
            return np.zeros(len(self.internships))
            
        scores = np.zeros(len(self.internships))
        user_skills_lower = {skill.lower(): i for i, skill in enumerate(user_skills, 1)}
        
        print(f"\n{'='*50}")
        print(f"Calculating skill similarity for user skills: {user_skills}")
        print(f"Skill index keys (first 10): {list(self.skill_index.keys())[:10]}")
        
        # Calculate base skill matches with position-based weighting
        for skill, weight in user_skills_lower.items():
            matching_indices = self.skill_index.get(skill, [])
            print(f"Skill '{skill}' found in {len(matching_indices)} internships")
            for idx in matching_indices:
                if 0 <= idx < len(scores):
                    # Higher weight for skills listed earlier (assuming more important)
                    score_increase = (1.0 / weight)
                    scores[idx] += score_increase
                    
                    # Debug output for the first few matches
                    if idx < 3:  # Only show for first few matches to avoid too much output
                        internship = self.internships[idx]
                        print(f"  - Match in internship {idx} (ID: {internship.get('id', 'N/A')}): {internship.get('title', 'N/A')}")
                        print(f"    Skills required: {internship.get('skills_required', [])}")
                        print(f"    Score increase: {score_increase:.4f}, New score: {scores[idx]:.4f}")
        
        # Normalize by the number of skills required
        print("\nFinal scores before normalization:")
        for idx, score in enumerate(scores):
            if score > 0:  # Only show non-zero scores
                internship = self.internships[idx]
                req_skills = len(internship.get('skills_required', []))
                print(f"  - Internship {idx} (ID: {internship.get('id', 'N/A')}): {score:.4f} (raw), {req_skills} required skills")
        
        # Apply normalization
        for idx, internship in enumerate(self.internships):
            req_skills = len(internship.get('skills_required', []))
            if req_skills > 0 and scores[idx] > 0:  # Only normalize if there are required skills and a score
                normalized_score = scores[idx] / min(len(user_skills_lower), req_skills)
                print(f"  - Normalizing internship {idx}: {scores[idx]:.4f} -> {normalized_score:.4f} (req_skills: {req_skills}, user_skills: {len(user_skills_lower)})")
                scores[idx] = normalized_score
        
        print(f"\nFinal scores after normalization (top 5):")
        top_indices = np.argsort(scores)[::-1][:5]
        for idx in top_indices:
            if scores[idx] > 0:
                print(f"  - Internship {idx}: {scores[idx]:.4f}")
        
        print("="*50 + "\n")
        return scores
    
    def _calculate_marks_similarity(self, user_marks: Dict[str, float]) -> np.ndarray:
        """Calculate scores based on how well user marks match internship requirements."""
        scores = np.ones(len(self.internships)) * 0.5  # Default score
        
        for idx, internship in enumerate(self.internships):
            # Check for CGPA requirements in description/requirements
            description = (internship.get('description', '') + ' ' + 
                         ' '.join(internship.get('requirements', []))).lower()
            
            # Look for CGPA requirements in the text
            cgpa_matches = re.findall(r'cumm?ulative gpa.*?([0-9]\.[0-9]|[0-9]\s*[+\-]?\s*[0-9]*)', description)
            if cgpa_matches:
                try:
                    required_cgpa = float(cgpa_matches[0])
                    user_cgpa = user_marks.get('cgpa', 0)
                    # Higher score for meeting or exceeding the requirement
                    if user_cgpa >= required_cgpa:
                        scores[idx] = 1.0
                    else:
                        # Partial score based on how close they are to meeting the requirement
                        scores[idx] = max(0.1, user_cgpa / required_cgpa)
                except (ValueError, TypeError):
                    pass
            
            # Check for percentage requirements
            percent_matches = re.findall(r'(?:minimum|min\.?|at least)?\s*([0-9]+)%', description)
            if percent_matches:
                try:
                    required_percent = float(percent_matches[0])
                    user_percent = user_marks.get('percentage', 0)
                    if user_percent >= required_percent:
                        scores[idx] = max(scores[idx], 1.0)  # Take the better score
                    else:
                        # Partial score based on how close they are to meeting the requirement
                        scores[idx] = max(scores[idx], max(0.1, user_percent / required_percent))
                except (ValueError, TypeError):
                    pass
        
        return scores
    
    def _calculate_skill_count_factor(self, user_skill_count: int) -> float:
        """Calculate a factor based on the number of skills the user has."""
        # This gives a boost to users with more skills, but with diminishing returns
        return min(1.5, 0.8 + (user_skill_count * 0.1))
    
    def _calculate_content_similarity(self, user_skills: List[str]) -> np.ndarray:
        """Calculate similarity scores based on TF-IDF vectors of descriptions."""
        if not user_skills or self.tfidf_matrix.shape[1] == 0:
            # Return neutral scores if no skills or empty TF-IDF matrix
            return np.ones(len(self.internships)) * 0.5
            
        try:
            # Create a query vector from user skills
            query_text = ' '.join(user_skills).lower()
            query_vector = self.vectorizer.transform([query_text])
            
            if query_vector.shape[1] != self.tfidf_matrix.shape[1]:
                # If dimensions don't match, return neutral scores
                print(f"Warning: Dimension mismatch in TF-IDF matrices: query={query_vector.shape}, matrix={self.tfidf_matrix.shape}")
                return np.ones(len(self.internships)) * 0.5
                
            query_vector = normalize(query_vector, axis=1)
            
            # Calculate cosine similarity with all internships
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)
            return similarities.flatten()
            
        except Exception as e:
            print(f"Error in content similarity calculation: {str(e)}")
            # Return neutral scores in case of error
            return np.ones(len(self.internships)) * 0.5
    
    def recommend_internships(
        self,
        user_skills: List[str],
        user_marks: Optional[Dict[str, float]] = None,
        skill_count: Optional[int] = None,
        top_n: int = 10,
        skill_weight: float = 0.5,
        content_weight: float = 0.3,
        marks_weight: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Generate internship recommendations based on user skills, marks, and skill count.
        
        Args:
            user_skills: List of skills the user possesses
            user_marks: Dictionary of user's marks (e.g., {'cgpa': 3.5, 'percentage': 85.0})
            skill_count: Number of skills the user has
            top_n: Number of recommendations to return
            skill_weight: Weight for skill-based matching (0-1)
            content_weight: Weight for content-based matching (0-1)
            marks_weight: Weight for marks/performance matching (0-1)
            
        Returns:
            List of recommended internships with relevance scores
        """
        if not user_skills or not self.internships:
            return []
            
        try:
            # Calculate base similarity scores
            skill_scores = self._calculate_skill_similarity(user_skills)
            
            # Make sure all score arrays have the same length as the number of internships
            num_internships = len(self.internships)
            
            # Initialize scores with neutral values
            content_scores = np.ones(num_internships) * 0.5
            marks_scores = np.ones(num_internships) * 0.5
            
            # Only calculate content scores if we have a valid TF-IDF matrix
            if hasattr(self, 'tfidf_matrix') and self.tfidf_matrix.shape[0] == num_internships:
                content_scores = self._calculate_content_similarity(user_skills)
            
            # Calculate marks-based scores if marks are provided
            if user_marks:
                marks_scores = self._calculate_marks_similarity(user_marks)
            
            # Calculate skill count factor if skill_count is provided
            skill_count_factor = 1.0
            if skill_count is not None:
                skill_count_factor = self._calculate_skill_count_factor(skill_count)
            
            # Combine scores with weights and skill count factor
            total_scores = (
                skill_weight * skill_scores * skill_count_factor +
                content_weight * content_scores * skill_count_factor +
                marks_weight * marks_scores
            )
            
            # Get top N recommendations
            top_indices = np.argsort(total_scores)[::-1][:top_n]
            
            # Prepare results
            recommendations = []
            for idx in top_indices:
                if idx < num_internships and total_scores[idx] > 0:  # Only include valid and relevant recommendations
                    try:
                        internship = self.internships[idx].copy()
                        if not isinstance(internship, dict):
                            continue
                            
                        internship['relevance_score'] = float(total_scores[idx])
                        internship['skill_match'] = float(skill_scores[idx] if idx < len(skill_scores) else 0)
                        internship['content_match'] = float(content_scores[idx] if idx < len(content_scores) else 0)
                        recommendations.append(internship)
                    except Exception as e:
                        print(f"Error processing internship at index {idx}: {str(e)}")
                        continue
            
            return recommendations
            
        except Exception as e:
            print(f"Error in recommend_internships: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_internship_by_id(self, internship_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an internship by its ID."""
        for internship in self.internships:
            if internship.get('id') == internship_id:
                return internship
        return None
    
    def get_recommendation_feedback(
        self, 
        user_skills: List[str], 
        liked_internships: List[str],
        disliked_internships: List[str]
    ) -> Dict[str, Any]:
        """Get feedback on why certain internships were recommended or not.
        
        Args:
            user_skills: List of user skills
            liked_internships: List of internship IDs the user liked
            disliked_internships: List of internship IDs the user disliked
            
        Returns:
            Dictionary containing feedback and suggestions
        """
        feedback = {
            'matched_skills': {},
            'missing_skills': {},
            'suggested_skills': set(),
            'recommended_domains': {}
        }
        
        # Analyze liked internships
        for int_id in liked_internships:
            internship = self.get_internship_by_id(int_id)
            if not internship:
                continue
                
            # Track matched skills
            for skill in internship.get('skills_required', []):
                if skill.lower() in {s.lower() for s in user_skills}:
                    feedback['matched_skills'][skill] = feedback['matched_skills'].get(skill, 0) + 1
            
            # Track recommended domains
            domain = internship.get('domain')
            if domain:
                feedback['recommended_domains'][domain] = feedback['recommended_domains'].get(domain, 0) + 1
        
        # Analyze disliked internships to find missing skills
        for int_id in disliked_internships:
            internship = self.get_internship_by_id(int_id)
            if not internship:
                continue
                
            for skill in internship.get('skills_required', []):
                if skill.lower() not in {s.lower() for s in user_skills}:
                    feedback['missing_skills'][skill] = feedback['missing_skills'].get(skill, 0) + 1
        
        # Suggest skills that appear frequently in missing skills
        if feedback['missing_skills']:
            top_missing = sorted(
                feedback['missing_skills'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            feedback['suggested_skills'] = [skill for skill, _ in top_missing]
        
        return feedback


def load_recommendation_engine() -> InternshipRecommendationEngine:
    """Helper function to load the recommendation engine with default settings."""
    return InternshipRecommendationEngine()


def get_recommendations(
    skills: List[str],
    marks: Optional[Dict[str, float]] = None,
    skill_count: Optional[int] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """Convenience function to get internship recommendations.
    
    Args:
        skills: List of skills to match against
        top_n: Number of recommendations to return
        
    Returns:
        List of recommended internships
    """
    engine = load_recommendation_engine()
    return engine.recommend_internships(
        user_skills=skills,
        user_marks=marks,
        skill_count=skill_count,
        top_n=top_n
    )
