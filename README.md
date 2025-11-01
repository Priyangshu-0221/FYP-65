# AI-Powered Resume Analyzer and Job Recommendation System

A comprehensive system for analyzing resumes and providing intelligent job recommendations based on skills, experience, and academic performance. The system processes PDF resumes, extracts key information, and uses machine learning to match candidates with relevant job opportunities.

## 🚀 Key Features

- **Resume Parsing**: Extract text, skills, education, and experience from PDF resumes
- **Skill Matching**: Advanced skill-based matching between candidates and job requirements
- **Academic Analysis**: Evaluate academic performance against job requirements
- **Smart Recommendations**: Personalized internship/job recommendations based on multiple factors
- **Web Interface**: User-friendly interface for uploading resumes and viewing matches
- **Batch Processing**: Process multiple resumes in one go

## 🏗️ Project Structure

```
FYP-65/
├── backend/                         # Backend Python code
│   ├── __init__.py
│   ├── fastapi_app.py              # Main FastAPI application
│   ├── pdf_processor.py            # PDF text extraction and processing
│   ├── recommendation_engine.py    # Recommendation system implementation
│   ├── create_dummy_recommendations.py  # Script to generate sample job data
│   ├── resume_classifier.py        # Resume classification model
│   ├── run.py                      # Script to run the FastAPI server
│   ├── schemas.py                  # Pydantic models
│   ├── settings.py                 # Application settings
│   └── skill_extractor.py          # Skill extraction utilities
├── data/                           # Data directory
│   └── dummy_internship_recommendations.json  # Sample job listings
├── frontend/                       # Frontend React application
│   ├── src/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── test_recommendation.py          # Test script for recommendation engine
├── requirements.txt                # Python dependencies
└── README.md
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Backend Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd FYP-65
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample job data** (if needed):
   ```bash
   python backend/create_dummy_recommendations.py
   ```

5. **Start the FastAPI server**:
   ```bash
   cd backend
   uvicorn fastapi_app:app --reload
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`

## 🚀 Usage

### API Endpoints

- `POST /upload-resume` - Upload and process a resume
- `POST /recommend` - Get job recommendations based on skills
- `GET /internships` - List all available internships

### Testing the Recommendation Engine

You can test the recommendation system using the provided test script:

```bash
python test_recommendation.py
```

This will run several test cases with different skill sets and display the recommendations.

### Example API Request

```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"skills": ["Python", "Machine Learning", "Data Analysis"], "top_k": 3}'
```

## 🤖 Recommendation Algorithm

The recommendation system uses a hybrid approach combining:

1. **Skill Matching**: Exact and fuzzy matching of skills with position-based weighting
2. **Content Similarity**: TF-IDF vectorization of job descriptions and requirements
3. **Academic Performance**: Matching of CGPA and percentage requirements

## 📊 Sample Output

```json
{
  "recommendations": [
    {
      "id": "ds-ml-001",
      "title": "Data Science Intern",
      "company": "Insight Analytics",
      "skills": ["python", "machine learning", "statistics", "sql"],
      "relevance_score": 0.95,
      "skill_match": 1.0,
      "content_match": 0.9
    },
    ...
  ]
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please open an issue or contact the project maintainers.
   - pip (Python package manager)

2. **Install Python Dependencies**
   ```bash
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set Up NLTK Data**
   ```python
   import nltk
   nltk.download(['punkt', 'stopwords', 'wordnet'])
   ```

## Usage

### 1. Process Resumes in Batch

To process all resumes in the `DATA/data/` directory and generate a CSV with extracted features:

```bash
python -m backend.process_resumes
```

This will create a `data/processed/resumes.csv` file with the extracted information.

### 2. Run the Web Interface

Start the FastAPI backend:

```bash
# From the project root
python -m backend.run
```

In a separate terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Access the web interface at `http://localhost:5173`

## API Endpoints

### POST /api/classify
Classify a resume and extract key information.

**Request Body**:
- `file`: PDF file to process

**Response**:
```json
{
  "category": "SOFTWARE_ENGINEER",
  "skills": ["Python", "Machine Learning", "Docker"],
  "education": "B.Tech in Computer Science (8.5/10)",
  "experience": "3 years"
}
```

## Configuration

Modify `backend/settings.py` to configure:
- Data paths
- Model parameters
- Feature extraction settings

## Data Format

The processed resumes are saved in CSV format with the following columns:
- `filepath`: Path to the original PDF
- `filename`: Name of the PDF file
- `category`: Job category (from directory name)
- `text_length`: Length of extracted text
- `total_skills`: Number of unique skills found
- `skills_found`: Comma-separated list of all skills
- `skills_frequency`: JSON string of skills with their counts
- `top_skills`: Top 5 most frequent skills
- `education_info`: Extracted education details
- `cgpa`: Extracted CGPA if found
- `percentage`: Extracted percentage if found
- `num_education`: Number of education entries
- `has_education`: 1 if education info found, else 0
- `has_marks`: 1 if marks/CGPA found, else 0
```

### Frontend (React + Chakra UI)

```bash
cd frontend
npm run dev
```

Features:

- Resume upload (PDF/TXT) with skill extraction display.
- Recommendation request based on extracted skills.
- Responsive cards showing internship matches with apply links.

---

## Command-line Utilities

Use these scripts for offline data preparation and experimentation:

### CSV Feature Engineering

```bash
python resume_feature_engineering.py \
  --input-csv DATA/resumes.csv \
  --output-csv processed_resumes.csv \
  --output-excel processed_resumes.xlsx \
  --skill-file skills.txt \
  --sample-rows 10
```

- `--skill-file` (optional) provides custom skills vocabulary.

### PDF Feature Extraction

```bash
python pdf_resume_feature_engineering.py \
  --pdf-dir DATA/pdf_resumes \
  --output-csv pdf_resumes_features.csv \
  --output-excel pdf_resumes_features.xlsx \
  --sample-rows 10
```

- PDF folder should be flat (no nested categories). Outputs include `Skills`, `Education`, `Marks` columns.

### Resume Classifier Training

```bash
python resume_classifier.py \
  --data-path DATA/resumes.csv \
  --model-type logistic_regression \
  --test-size 0.2 \
  --predict-text "Senior data analyst with Python and SQL experience"
```

- Supports `naive_bayes` and `logistic_regression` models.
- Outputs accuracy, classification report, confusion matrix, and optional inference.

---

## Customization & Extension

- **Skill Vocabulary**: Update `DEFAULT_SKILLS` in `resume_feature_engineering.py` or `web_backend/backend.py`. Supply a text file of skills for CLI usage.
- **Internship Catalog**: Override the default catalog by providing a JSON file and pointing `INTERNSHIP_APP_INTERNSHIP_CATALOG_PATH` to it.
- **Recommendation Logic**: Modify `web_backend/backend.py` to implement weighted scoring, ML-based ranking, or incorporate category predictions.
- **Model Enhancements**: Extend `resume_classifier.py` with additional algorithms or hyperparameter tuning.

---

## Troubleshooting

- **Missing PDF Text**: Some PDFs lack extractable text. Consider OCR preprocessing.
- **NLTK Lookup Errors**: Ensure corpora download successfully or set `NLTK_DATA` to a writable location.
- **Frontend API Errors**: Confirm the backend is running on `http://127.0.0.1:8000` and the proxy is configured correctly.
- **npm Issues**: Delete `frontend/node_modules` and rerun `npm install`. Ensure Node 18+ is installed.

---

