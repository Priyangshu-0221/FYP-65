<div align="center">

# 🧠 AI Powered Career Guidance System - Backend

> **Resume Processing Engine & Internship Recommendation API**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img alt="AI/ML" src="https://img.shields.io/badge/AI%2FML-Sentence%20Transformers-FF6B35?style=for-the-badge" />
  <img alt="NLP" src="https://img.shields.io/badge/NLP-PyPDF2-412991?style=for-the-badge" />
</p>

🎓 _Bachelor of Technology (CSE) Final Year Project_  
📍 _Sister Nivedita University, 2025_

</div>

---

## 📖 About

The **Backend API** is a high-performance, production-ready system that powers the AI Career Guidance platform. Built with **FastAPI**, it provides intelligent resume processing, skill extraction, and semantic-based internship recommendations using advanced NLP techniques. The engine leverages state-of-the-art **Sentence Transformers** for semantic similarity matching to deliver highly relevant internship matches.

---

## ✨ Key Features

### 📄 Resume Processing Pipeline

- **PDF Text Extraction** – Accurate parsing of resume PDFs using PyPDF2
- **Skill Recognition** – 500+ technical and professional skills identification
- **Education Extraction** – Degree, university, and CGPA/percentage detection
- **Experience Analysis** – Work history, roles, and responsibilities extraction
- **Multi-Domain Support** – Technology, finance, HR, healthcare, engineering sectors

### 🤖 AI-Powered Internship Recommendations

| Feature                | Details                                      | Technology                    |
| ---------------------- | -------------------------------------------- | ----------------------------- |
| **Semantic Matching**  | Match skills to internships using embeddings | Sentence Transformers         |
| **Scoring System**     | Relevance scores for ranking results         | Cosine Similarity             |
| **Flexible Models**    | Switch between 3 industry-standard models    | BAAI/bge-small, MiniLM, MPNet |
| **Skill Gap Analysis** | Identify missing skills for target roles     | NLP-based comparison          |
| **Fallback Logic**     | Graceful degradation to basic matching       | Always works, never fails     |

### 🔧 Production-Ready Infrastructure

- 🔐 **CORS Support** – Secure cross-origin requests from frontend
- 📊 **Health Monitoring** – Endpoint availability checks
- ⚡ **Async Processing** – Non-blocking file uploads and recommendations
- 🛡️ **Error Handling** – Comprehensive exception management
- 📝 **Logging** – Request/response tracking for debugging

### 📚 Internship Database

- **25+ Internship Opportunities** – Pre-loaded dummy dataset
- **Multiple Sectors** – Tech, Finance, HR, Design, Healthcare, Engineering
- **Global Locations** – Remote, India, UK, Singapore, USA-based roles
- **Rich Metadata** – Company, location, skills, descriptions, apply links

---

## 🛠️ Tech Stack

<div align="center">

| Layer            | Technology            | Purpose                       |
| ---------------- | --------------------- | ----------------------------- |
| **🌐 Framework** | FastAPI 0.100+        | Modern async web framework    |
| **🐍 Language**  | Python 3.10+          | Data science & ML ecosystem   |
| **🧠 AI/ML**     | Sentence Transformers | Semantic embeddings           |
| **📄 PDF**       | PyPDF2                | PDF text extraction           |
| **📊 ML Ops**    | Scikit-learn          | Similarity computation        |
| **🔢 Compute**   | NumPy, PyTorch        | Numerical & tensor operations |
| **⚙️ Config**    | Pydantic Settings     | Environment configuration     |
| **🔀 CORS**      | FastAPI Middleware    | Cross-origin request handling |

</div>

---

## 📦 Dependencies

```
fastapi==0.104.1              # Web framework
uvicorn==0.24.0               # ASGI server
python-multipart==0.0.6       # Form data parsing
PyPDF2==3.17.1                # PDF processing
sentence-transformers==2.2.2  # Semantic embeddings
scikit-learn==1.3.2           # ML algorithms
numpy==1.24.3                 # Numerical computing
torch==2.0.1                  # PyTorch (auto-installed with transformers)
pydantic==2.0+                # Data validation
pydantic-settings==2.0+       # Config management
```

---

## 📂 Project Structure

```
backend/
│
├── 📄 app.py                 # FastAPI application & endpoints
├── 📄 extractor.py           # Resume PDF extraction logic
├── 📄 recommendation_engine.py # AI/ML semantic matching engine
├── 📄 internship_catalog.py   # Internship data management
├── 📄 schemas.py             # Pydantic data models
├── 📄 settings.py            # Application settings & config
├── 📄 __init__.py            # Package initialization
├── 📄 __main__.py            # Entry point for running server
│
├── 📁 data/                  # Runtime data storage
│   └── dummy_internship_recommendations.json
│
├── 📁 __pycache__/           # Compiled Python bytecode
│
└── README.md                 # This file
```

### Core Module Documentation

| File                         | Responsibility                      | Key Functions                                            |
| ---------------------------- | ----------------------------------- | -------------------------------------------------------- |
| **app.py**                   | FastAPI initialization & API routes | `/health`, `/upload`, `/recommend`                       |
| **extractor.py**             | Resume text & metadata extraction   | `process_resume()`, skill/education detection            |
| **recommendation_engine.py** | Semantic similarity & ranking       | `recommend_internships()`, `get_skill_recommendations()` |
| **internship_catalog.py**    | Database & query operations         | Catalog loading, filtering, ranking                      |
| **schemas.py**               | Type definitions                    | Request/response validation                              |
| **settings.py**              | Configuration management            | Environment variables, paths                             |

---

## 🚀 Getting Started

### 📋 Prerequisites

```bash
✓ Python 3.10+           (3.11 or 3.12 recommended)
✓ pip (Python package manager)
✓ Virtual Environment (recommended: venv or conda)
✓ 2GB+ RAM              (for Sentence Transformers models)
✓ Internet Connection   (for first-time model downloads)
```

### 💻 Installation

**1️⃣ Navigate to backend directory:**

```bash
cd backend
```

**2️⃣ Create and activate virtual environment:**

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**3️⃣ Install dependencies:**

```bash
pip install -r requirements.txt
```

> **Note:** First time installation downloads AI models (~200-400MB). Ensure stable internet connection.

**4️⃣ Verify installation:**

```bash
python -c "from sentence_transformers import SentenceTransformer; print('✅ All packages installed!')"
```

**5️⃣ Start the backend server:**

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080 --reload
```

**6️⃣ Access the API:**

- 🌐 **Web API:** [http://localhost:8080](http://localhost:8080)
- 📚 **Swagger UI:** [http://localhost:8080/docs](http://localhost:8080/docs)
- 🔄 **ReDoc API Docs:** [http://localhost:8080/redoc](http://localhost:8080/redoc)

---

## 📝 Available Commands

| Command                                               | Description               | Purpose                   |
| ----------------------------------------------------- | ------------------------- | ------------------------- |
| `python -m backend`                                   | Run with default settings | Production-like execution |
| `python -m uvicorn backend.app:app --reload`          | Hot-reload development    | Development workflow      |
| `python -m uvicorn backend.app:app --port 8000`       | Custom port               | Alternative port binding  |
| `python -m uvicorn backend.app:app --log-level debug` | Verbose logging           | Debugging issues          |

---

## 🔌 API Documentation

### 1️⃣ Health Check Endpoint

**Endpoint:** `GET /health`

**Purpose:** Verify backend availability

**Response:**

```json
{
  "status": "ok",
  "message": "Server is running"
}
```

**Usage:**

```bash
curl http://localhost:8080/health
```

---

### 2️⃣ Resume Upload & Processing

**Endpoint:** `POST /upload`

**Purpose:** Upload PDF resume, extract skills, education, experience

**Request:**

- **Method:** POST
- **Content-Type:** multipart/form-data
- **Parameter:** `file` (PDF file)

**Response:**

```json
{
  "success": true,
  "text": "Extracted resume text (truncated to 1000 chars)...",
  "skills": ["python", "java", "react", "sql"],
  "category": "General",
  "education": ["B.Tech in Computer Science"],
  "experience": ["Software Developer at Tech Corp"],
  "skill_count": 4,
  "cgpa": 8.5,
  "percentage": 85
}
```

**Example:**

```bash
curl -X POST -F "file=@resume.pdf" http://localhost:8080/upload
```

**Error Responses:**

- **400 Bad Request** – Non-PDF file uploaded
- **400 Bad Request** – Empty file
- **500 Internal Server Error** – PDF processing failed

---

### 3️⃣ Internship Recommendations

**Endpoint:** `POST /recommend`

**Purpose:** Get AI-powered internship recommendations based on skills

**Request Body:**

```json
{
  "skills": ["python", "machine learning", "data analysis"],
  "marks": {
    "cgpa": 8.5,
    "percentage": 85,
    "sem_scores": [90, 88, 85, 87]
  },
  "top_k": 6
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "id": "5",
      "title": "AI/ML Intern",
      "company": "Neural Hub",
      "location": "Hyderabad, India",
      "category": "AI",
      "skills": ["python", "tensorflow", "pytorch", "scikit-learn"],
      "description": "Assist in building next-gen ML models.",
      "apply_link": "#",
      "match_score": 0.92,
      "relevance": "Excellent Match"
    }
  ],
  "recommended_skills": [
    {
      "skill": "tensorflow",
      "importance": 0.95,
      "category": "Deep Learning"
    }
  ]
}
```

**Parameters:**

- `skills` (required): Array of user skills (lowercase)
- `marks` (optional): Academic performance data
- `top_k` (optional): Number of recommendations (default: 6)

**Error Responses:**

- **400 Bad Request** – No skills provided
- **500 Internal Server Error** – Recommendation engine failed

---

### 📊 API Response Status Codes

| Code    | Status                | Meaning                                |
| ------- | --------------------- | -------------------------------------- |
| **200** | OK                    | Request successful, data returned      |
| **400** | Bad Request           | Invalid input, missing required fields |
| **500** | Internal Server Error | Server processing error                |

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Backend Server
INTERNSHIP_APP_BACKEND_HOST=0.0.0.0
INTERNSHIP_APP_BACKEND_PORT=8080

# Data Paths
INTERNSHIP_APP_RESUME_DATASET_PATH=DATA/data
INTERNSHIP_APP_INTERNSHIP_CATALOG_PATH=DATA/dummy_internship_recommendations.json

# Recommendation Engine (Optional)
INTERNSHIP_APP_MODEL_NAME=BAAI/bge-small-en-v1.5
INTERNSHIP_APP_LOG_LEVEL=info
```

### Recommendation Models

The engine supports 3 models for different use cases:

| Model         | Key         | Best For                        | Speed  | Accuracy   |
| ------------- | ----------- | ------------------------------- | ------ | ---------- |
| **MiniLM**    | `minilm`    | Fast inference, lower latency   | ⚡⚡⚡ | ⭐⭐⭐     |
| **BGE-Small** | `bge-small` | Balanced, research-backed       | ⚡⚡   | ⭐⭐⭐⭐⭐ |
| **MPNet**     | `mpnet`     | Maximum accuracy, comprehensive | ⚡     | ⭐⭐⭐⭐   |

**Switch Models in `recommendation_engine.py`:**

```python
ACTIVE_MODEL_KEY = "bge-small"  # Change this to switch
```

---

## 🔄 Workflow Explanation

### Resume Processing Flow

```
PDF File Upload
      ↓
  [File Validation]
      ↓
  [PDF Text Extraction]
      ↓
  [Skill Recognition] → Extracts relevant skills
      ↓
  [Education Detection] → Finds degree, CGPA, university
      ↓
  [Experience Parsing] → Identifies job roles
      ↓
  [Academic Score Detection] → Extracts GPA/percentage
      ↓
  [JSON Response] → Sends to frontend
```

### Recommendation Generation Flow

```
User Skills Input
      ↓
  [Skill Embedding] → Convert to vector using Sentence Transformers
      ↓
  [Similarity Computation] → Calculate cosine similarity to all internships
      ↓
  [Score Ranking] → Sort by relevance score
      ↓
  [Top-K Selection] → Select top 6 matches
      ↓
  [Skill Gap Analysis] → Find missing skills
      ↓
  [JSON Response] → Sends to frontend
```

---

## 🧪 Development Workflow

### Running in Development Mode

```bash
# With auto-reload on file changes
python -m uvicorn backend.app:app --reload --port 8080

# With debug logging
python -m uvicorn backend.app:app --reload --log-level debug
```

### Testing an Endpoint

**Using Python requests:**

```python
import requests
import json

# Test health
response = requests.get("http://localhost:8080/health")
print(response.json())

# Test recommendations
payload = {
    "skills": ["python", "data analysis", "sql"],
    "marks": {"cgpa": 8.5},
    "top_k": 6
}
response = requests.post(
    "http://localhost:8080/recommend",
    json=payload
)
print(json.dumps(response.json(), indent=2))
```

### Debugging Tips

| Issue                        | Solution                                                                                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Port 8080 already in use** | Change port: `--port 8081`                                                                                                                   |
| **PDF extraction fails**     | Ensure PDF is valid, try different PDF                                                                                                       |
| **Model download hangs**     | Check internet, manually: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"` |
| **Skill not recognized**     | Add to SKILLS set in `extractor.py`                                                                                                          |
| **Slow recommendations**     | Switch to `minilm` model for speed                                                                                                           |

---

## 📱 Frontend Integration

### API Base URL Configuration

In frontend `src/services/api.ts`:

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
```

### Example Frontend Usage

```typescript
// Upload resume
const formData = new FormData();
formData.append("file", resumeFile);
const uploadResponse = await fetch(`${API_BASE}/upload`, {
  method: "POST",
  body: formData,
});
const { skills, education, cgpa, percentage } = await uploadResponse.json();

// Get recommendations
const recommendResponse = await fetch(`${API_BASE}/recommend`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    skills: skills.map((s) => s.toLowerCase()),
    marks: { cgpa, percentage },
    top_k: 6,
  }),
});
const { recommendations, recommended_skills } = await recommendResponse.json();
```

---

## 🔒 Security Considerations

- ✅ **File Upload Validation** – Only PDF files accepted
- ✅ **File Size Limits** – Recommended < 10MB
- ✅ **CORS Enabled** – All origins allowed (for development)
- ✅ **Error Masking** – Sensitive errors don't leak to client
- ✅ **Temp File Cleanup** – Uploaded files deleted after processing

### Production Security

For production deployment:

```python
# Restrict CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific frontend URL
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Add file size validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'sentence_transformers'`**

```bash
Solution: pip install sentence-transformers
```

**Issue: `Port 8080 is already in use`**

```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/macOS
lsof -i :8080
kill -9 <PID>
```

**Issue: `PDF extraction returns empty text`**

- Verify PDF is not image-based (scanned document)
- Try a different PDF file
- Check PDF is not password protected

**Issue: Slow recommendation generation**

- Switch to faster model: `minilm`
- Reduce `top_k` parameter
- Check system RAM (needs ~2GB for BGE model)

**Issue: `ModuleNotFoundError: No module named 'backend'`**

```bash
# Run from project root, not from backend folder
cd ..
python -m uvicorn backend.app:app --port 8080
```

---

## 📚 API Request Examples

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# Upload resume
curl -X POST -F "file=@resume.pdf" http://localhost:8080/upload

# Get recommendations
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["python", "data analysis"],
    "marks": {"cgpa": 8.5},
    "top_k": 6
  }'
```

### Python Examples

```python
import requests

# Health check
resp = requests.get("http://localhost:8080/health")
print(resp.json())

# Upload & process resume
with open("resume.pdf", "rb") as f:
    resp = requests.post("http://localhost:8080/upload", files={"file": f})
    print(resp.json())

# Get internship recommendations
payload = {
    "skills": ["python", "machine learning", "data analysis"],
    "marks": {"cgpa": 8.5},
    "top_k": 6
}
resp = requests.post("http://localhost:8080/recommend", json=payload)
print(resp.json())
```

---

## 🎓 Project Information

| Detail         | Information                                                     |
| -------------- | --------------------------------------------------------------- |
| **University** | Sister Nivedita University (SNU)                                |
| **Program**    | Bachelor of Technology (BTech) - Computer Science & Engineering |
| **Supervisor** | Dr. Sayani Mondal (Assistant Professor)                         |
| **Submission** | November 25, 2025                                               |
| **Session**    | Academic Year 2025                                              |

### Developed By

| Name              | Registration | Email                        |
| ----------------- | ------------ | ---------------------------- |
| Priyangshu Mondal | 220100663543 | mondalpriyangshu@gmail.com   |
| Abhijit Biswas    | 220100017663 | abhijit.biswas1024@gmail.com |
| Kunal Roy         | 220100185465 | royku321@gmail.com           |
| Rupam Haldar      | 220100408950 | prabirhaldar68@gmail.com     |

---

## 📄 License

This project is created as part of an academic Final Year Project and is intended for educational purposes.

---

<p align="center">
  <strong>Built with 🤖 using FastAPI, Python, and Sentence Transformers</strong>
  <br />
  <em>AI Powered Career Guidance System - Backend API Server</em>
</p>
