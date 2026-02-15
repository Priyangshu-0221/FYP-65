# AI-Powered Internship Recommendation System

A state-of-the-art system that uses **Artificial Intelligence** and **Semantic Search** to match students with internships. Unlike simple keyword matching, this system understands the *meaning* of skills (e.g., knowing "Machine Learning" implies a fit for "Data Science" roles).

## 🚀 Key Features

- **🧠 Semantic Search Engine**: Uses `Sentence-Transformers` (Deep Learning) to understand the context of resumes and job descriptions.
- **📄 Smart Resume Parsing**: Extracts Skills, Education, and Experience from PDF/Text resumes.
- **📊 Excel Reporting Pipeline**: Automatically generates detailed reports (`raw_data.xlsx`, `cleaned_data.xlsx`, `tokenized_data.xlsx`) for every upload to visualize the AI's processing.
- **🎓 Academic Filtering**: Considers CGPA/Percentage as a secondary filter for eligibility.
- **💻 Modern UI**: A beautiful, animated React frontend for a premium user experience.

## 🏗️ Project Structure

```
FYP-65/
├── backend/
│   ├── data/
│   │   └── dummy_internship_recommendations.json  # Internship Database
│   ├── reports/                    # Excel logs generated here
│   ├── fastapi_app.py              # Main API Server
│   ├── recommendation_engine.py    # AI Semantic Search Logic
│   ├── pdf_processor.py            # PDF Text Extraction
│   ├── skill_extractor.py          # NLP Utilities
│   ├── schemas.py                  # API Data Models
│   ├── settings.py                 # Config
│   └── requirements.txt            # Python Dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Main UI Logic
│   │   └── main.jsx
│   └── ...
└── README.md
```

## 🛠️ Setup Instructions

### 1. Backend Setup
The backend uses Python and powerful AI models.

```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt

# Start Server
python run.py
```
*Note: The first run will download the `all-MiniLM-L6-v2` AI model (~100MB). This happens only once.*

### 2. Frontend Setup
The frontend is built with React and Vite.

```bash
cd frontend

# Install Dependencies
npm install

# Start UI
npm run dev
```
Access the app at `http://localhost:5173` (or the port shown in terminal).

## 📝 How It Works

1.  **Upload**: User uploads a Resume (PDF).
2.  **Process**:
    *   Text is extracted and cleaned.
    *   Data is logged to `backend/reports/` for audit.
    *   Skills are extracted using NLP.
3.  **Embed**: The Resume's skills and the Internship descriptions are converted into **Vector Embeddings** (numbers representing meaning).
4.  **Match**: We calculate the **Cosine Similarity** between the resume vector and internship vectors.
5.  **Rank**: Jobs are ranked by Semantic Score (70%) + Keyword Match (20%) + Academic Fit (10%).

## 📊 Data Pipeline Reports

Check the `backend/reports/` folder after uploading a resume to see:
*   `raw_data.xlsx`: Original text.
*   `cleaned_data.xlsx`: Pre-processed text.
*   `tokenized_data.xlsx`: Tokenized words used by the model.

## 🔗 API Endpoints

- `POST /upload`: Upload a resume file. Returns extracted skills.
- `POST /recommend`: Get top internship matches based on skills and marks.
