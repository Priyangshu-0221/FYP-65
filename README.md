# 🧠 Smart CV Analyzer: AI-Powered Internship Matching

A cutting-edge platform that uses **Semantic Artificial Intelligence** to bridge the gap between students and their dream internships. Unlike traditional systems that rely on simple keyword matching, Smart CV Analyzer understands the *context* and *meaning* behind a student's skills to find the perfect career fit.

---

## 🌟 Why This Project?

Finding the right internship is often a chaotic process:
*   **Students** struggle to identify which roles fit their specific skill set.
*   **Recruiters** are overwhelmed by resumes that don't match the job description.
*   **Traditional Portals** use "exact match" keywords (e.g., searching for "programmer" misses "developer"), leading to missed opportunities.

**Smart CV Analyzer solves this by thinking like a human recruiter.** It reads a resume, understands that "Python" and "Data Analysis" makes you a good fit for "Data Science" (even if the term "Data Science" isn't in your CV), and recommends the best opportunities.

---

## 🚀 How It Works (The AI Pipeline)

The system follows a sophisticated 5-step pipeline to transform a raw PDF resume into actionable career advice:

### 1. **Resume Processing (OCR & NLP)**
*   **Input**: User uploads a PDF/DOCX resume.
*   **Extraction**: The system uses `pdf_processor` to extract raw text.
*   **Cleaning**: Text is cleaned (removing special characters, formatting) and tokenized.
*   **Entity Extraction**: We use **Natural Language Processing (NLP)** to identify key entities:
    *   **Skills** (e.g., Python, React, AWS)
    *   **Education** (Degree, College)
    *   **Experience** (Years, Roles)
    *   **Academic Score** (CGPA/Percentage)

### 2. **Semantic Understanding (The "Brain")**
*   **Model**: We use `Sentence-Transformers` (specifically `all-MiniLM-L6-v2`), a deep learning model pre-trained on millions of data points.
*   **Vector Embeddings**:
    *   The model converts the student's *skills profile* into a **high-dimensional vector** (a list of 384 numbers).
    *   It also converts every *internship description* in our database into similar vectors.
*   **Contextual Match**: These vectors represent the *meaning* of the text. So, "Machine Learning" and "Artificial Intelligence" will have vectors that are numerically very close, allowing the system to match them even if the words are different.

### 3. **The Recommendation Engine**
*   **Cosine Similarity**: We calculate the mathematical similarity (angle) between the **Student Vector** and every **Internship Vector**.
*   **Hybrid Scoring**: The final score is not just AI. It's a robust weighted average:
    *   **Semantic Score (70%)**: How well does the *meaning* match?
    *   **Keyword Match (20%)**: Do they have the specific required hard skills?
    *   **Academic Fit (10%)**: Do they meet the CGPA criteria?

### 4. **Transparency & Reporting**
*   **Audit Trail**: To ensure trust, the system logs every step of the data processing into Excel reports (`backend/reports/`):
    *   `raw_data.xlsx`: What we read.
    *   `cleaned_data.xlsx`: How we cleaned it.
    *   `tokenized_data.xlsx`: How the AI sees it.

### 5. **User Interface**
*   A modern, responsive React frontend provides a seamless experience:
    *   **Real-time Analysis**: See extracted skills instantly.
    *   **Interactive Recommendations**: View job cards with similarity scores.
    *   **Direct Application**: Apply to matched internships with one click.

---

## 🏗️ Technical Architecture

### **Backend (Python & FastAPI)**
*   **FastAPI**: High-performance web framework for the API.
*   **PyTorch & Sentence-Transformers**: The core AI/Deep Learning engine.
*   **Scikit-Learn**: For additional data processing utilities.
*   **Pandas & OpenPyXL**: For data manipulation and Excel reporting.

### **Frontend (React)**
*   **Vite**: Next-generation frontend tooling for speed.
*   **Chakra UI**: Component library for accessible, professional design.
*   **Framer Motion**: For smooth, engaging animations.

---

## 🛠️ Setup & Installation

### Prerequisites
*   Python 3.10+
*   Node.js 16+

### 1. Backend Setup
```bash
cd backend
# Create virtual environment (Recommended)
python -m venv venv
# Activate: venv\Scripts\activate (Win) or source venv/bin/activate (Mac/Linux)

# Install Dependencies (Note: This downloads PyTorch ~2GB)
pip install -r requirements.txt

# Run the API
python run.py
```
*The server will start at `http://localhost:8000`.*

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*Access the UI at `http://localhost:5173`.*

---

## 📊 Data & Logs
*   **Internship Database**: `backend/data/dummy_internship_recommendations.json` (Editable JSON).
*   **Processing Logs**: Check `backend/reports/` after uploading a resume.

---

## 🔮 Future Roadmap
*   **Cover Letter Generator**: Auto-generate cover letters based on the matched internship.
*   **Mock Interviewer**: AI chatbot to practice interview questions for the specific role.
*   **Resume Scorer**: Give a score (0-100) on resume quality.

---
**Developed for Final Year Project (FYP-65)**
