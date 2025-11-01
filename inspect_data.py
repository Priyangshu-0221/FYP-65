import json
from pathlib import Path

def inspect_data():
    data_path = Path("data/dummy_internship_recommendations.json")
    
    if not data_path.exists():
        print(f"Error: {data_path} does not exist")
        return
    
    print(f"Loading data from {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print("Dictionary keys:", list(data.keys()))
            if 'internships' in data:
                print(f"Found 'internships' key with {len(data['internships'])} items")
                if len(data['internships']) > 0:
                    first_item = data['internships'][0]
                    print("\nFirst internship item type:", type(first_item))
                    print("First internship keys:", list(first_item.keys()) if isinstance(first_item, dict) else "N/A")
                    print("First internship sample:", str(first_item)[:200] + "...")
        elif isinstance(data, list):
            print(f"Data is a list with {len(data)} items")
            if len(data) > 0:
                first_item = data[0]
                print("\nFirst item type:", type(first_item))
                if isinstance(first_item, dict):
                    print("First item keys:", list(first_item.keys()))
                    print("First item sample:", str(first_item)[:200] + "...")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    inspect_data()
