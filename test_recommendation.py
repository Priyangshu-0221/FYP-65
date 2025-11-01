import requests
import json
from typing import Dict, Any, List

def test_recommendation(skills: List[str], top_k: int = 5, marks: Dict[str, float] = None) -> Dict[str, Any]:
    """Test the recommendation endpoint with the given skills and marks."""
    # Test data
    test_data = {
        "skills": skills,
        "top_k": top_k
    }
    
    # Add marks if provided
    if marks:
        test_data["marks"] = marks
    
    print("\n" + "="*50)
    print(f"Testing with skills: {skills}")
    if marks:
        print(f"Marks: {marks}")
    print("-"*50)
    
    try:
        # Send request to the recommendation endpoint
        response = requests.post(
            "http://localhost:8000/recommend",
            json=test_data,
            timeout=10
        )
        
        # Print the response status and data
        print(f"Status Code: {response.status_code}")
        
        try:
            response_data = response.json()
            print("Response:")
            print(json.dumps(response_data, indent=2))
            return response_data
            
        except json.JSONDecodeError:
            print("Error: Response is not valid JSON")
            print(f"Response text: {response.text[:500]}...")
            return {"error": "Invalid JSON response", "text": response.text}
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return {"error": f"Request failed: {str(e)}"}

def main():
    # Test cases
    test_cases = [
        {
            "name": "Basic technical skills",
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "marks": {"cgpa": 3.5, "percentage": 85.0},
            "top_k": 5
        },
        {
            "name": "Web development",
            "skills": ["JavaScript", "React", "Node.js", "HTML", "CSS"],
            "marks": {"cgpa": 3.2, "percentage": 82.0},
            "top_k": 3
        },
        {
            "name": "Data Science",
            "skills": ["Python", "R", "SQL", "Pandas", "TensorFlow"],
            "marks": {"cgpa": 3.8, "percentage": 90.0},
            "top_k": 4
        },
        {
            "name": "Minimal skills",
            "skills": ["Python"],
            "top_k": 2
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} Test Case {i}: {test_case['name']} {'='*20}")
        test_recommendation(
            skills=test_case['skills'],
            top_k=test_case['top_k'],
            marks=test_case.get('marks')
        )

if __name__ == "__main__":
    main()
