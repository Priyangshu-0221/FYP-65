# 🎓 Smart CV Analyzer - AI-Powered Internship Recommendation System

An intelligent resume analysis and internship matching system that extracts skills, education, and experience from PDF resumes and provides personalized internship recommendations using AI.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

---

## 🌟 Features

- **📄 PDF Resume Extraction**: Automatically extracts text from PDF resumes using PyPDF2
- **🎯 Skill Detection**: Identifies 100+ technical and soft skills from resume text
- **🎓 Education & Experience Parsing**: Extracts education details, work experience, CGPA, and percentage
- **📊 Dataset Processing**: Batch process thousands of resumes and export to Excel (raw, cleaned, tokenized)
- **🤖 AI Recommendations**: Intelligent internship matching based on extracted skills
- **💎 Modern UI**: Beautiful, responsive React frontend with real-time updates and animations

---

## 📁 Project Structure

```
FYP-65/
├── backend/                          # Backend package
│   ├── app.py                        # Main FastAPI application
│   ├── extractor.py                  # PDF extraction & skill detection
│   ├── recommendation_engine.py      # AI recommendation logic
│   ├── internship_catalog.py         # Internship data loader
│   ├── schemas.py                    # Pydantic data models
│   ├── settings.py                   # Configuration settings
│   ├── __init__.py                   # Package initialization
│   ├── __main__.py                   # Entry point
│   └── data/
│       └── dummy_internship_recommendations.json
│
├── frontend/                         # React frontend application
│   ├── src/
│   │   ├── App.jsx                   # Main React component
│   │   └── styles/
│   ├── vite.config.js                # Vite configuration
│   └── package.json                  # Node dependencies
│
├── DATA/                             # Resume dataset
│   ├── data/                         # Resume PDFs (24 categories)
│   │   ├── ACCOUNTANT/
│   │   ├── ENGINEERING/
│   │   ├── INFORMATION-TECHNOLOGY/
│   │   └── ... (21 more categories)
│   └── processed/                    # Generated Excel files
│       ├── raw_data.xlsx             # Original text + extracted info
│       ├── cleaned_data.xlsx         # Cleaned text
│       └── tokenized_data.xlsx       # Tokenized text
│
├── scripts/                          # Utility scripts
│   └── process_all_resumes.py        # Dataset batch processor
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.8 or higher
- **Node.js** 16 or higher
- **pip** (Python package manager)
- **npm** (Node package manager)

### Installation

1. **Clone the repository**
   ```bash
   cd FYP-65
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

**Terminal 1 - Start Backend:**
```bash
python -m backend
```
Backend will run on `http://localhost:8000`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```
Frontend will run on `http://localhost:5173`

**Open your browser and go to:** `http://localhost:5173`

---

## 📊 Dataset Processing

Process your entire resume dataset and generate Excel files:

```bash
python scripts/process_all_resumes.py
```

This creates 3 Excel files in `DATA/processed/`:

1. **raw_data.xlsx**: Original text + all extracted information
   - Columns: filename, category, text, skills, skills_count, education, experience, cgpa, percentage

2. **cleaned_data.xlsx**: Cleaned text (lowercase, no special characters)
   - Columns: filename, category, cleaned_text, skills, skills_count

3. **tokenized_data.xlsx**: Tokenized words for NLP analysis
   - Columns: filename, category, tokenized_text, token_count, skills

**Processing speed:** ~6-8 resumes per second

---

## 🔧 API Endpoints

### `POST /upload`
Upload a resume PDF and extract information.

**Request:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "success": true,
  "text": "Resume text...",
  "skills": ["python", "java", "sql", "react", "docker"],
  "education": "B.Tech Computer Science, XYZ University",
  "experience": "Software Engineer at ABC Corp (2 years)",
  "cgpa": "8.5",
  "percentage": "85"
}
```

### `POST /recommend`
Get internship recommendations based on skills.

**Request:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "skills": ["python", "react", "sql"],
    "top_k": 6,
    "marks": {"cgpa": "8.5", "percentage": "85"},
    "skill_count": 3
  }'
```

**Response:**
```json
{
  "recommendations": [
    {
      "id": "1",
      "title": "Software Developer Intern",
      "company": "Tech Corp",
      "location": "Remote",
      "category": "IT",
      "skills": ["python", "react"],
      "description": "Great opportunity for developers...",
      "apply_link": "https://..."
    }
  ]
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "Server is running"
}
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast Python web framework
- **PyPDF2**: PDF text extraction
- **pandas**: Data processing and Excel export
- **openpyxl**: Excel file handling
- **uvicorn**: ASGI server

### Frontend
- **React**: UI framework
- **Vite**: Build tool and dev server
- **Chakra UI**: Component library
- **React Icons**: Icon library

### Data Processing
- **tqdm**: Progress bars for batch processing
- **Regular Expressions**: Pattern matching for skill detection

---

## 📝 Skills Detected

The system detects **100+ skills** including:

**Programming Languages:**
Python, Java, C++, JavaScript, TypeScript, SQL, R, Go, Rust, Kotlin, Swift, PHP, Ruby, Scala, etc.

**Web Technologies:**
React, Angular, Vue, Node.js, Django, Flask, FastAPI, HTML, CSS, Bootstrap, Tailwind, etc.

**Databases:**
MySQL, PostgreSQL, MongoDB, Redis, Cassandra, Oracle, SQL Server, etc.

**Cloud & DevOps:**
AWS, Azure, GCP, Docker, Kubernetes, Jenkins, CI/CD, Terraform, etc.

**Data Science & ML:**
Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NumPy, etc.

**Tools & Frameworks:**
Git, GitHub, Linux, Agile, Scrum, REST API, GraphQL, etc.

**Soft Skills:**
Leadership, Communication, Teamwork, Problem Solving, Time Management, etc.

---

## 🎯 How It Works

1. **Upload Resume**: User uploads a PDF resume through the web interface
2. **Text Extraction**: PyPDF2 extracts text from all pages of the PDF
3. **Skill Detection**: Pattern matching identifies skills from a predefined list of 100+ skills
4. **Information Extraction**: Regular expressions extract education, experience, CGPA, and percentage
5. **Display Results**: Frontend displays extracted skills with beautiful animations
6. **Get Recommendations**: User can request internship recommendations
7. **AI Matching**: Recommendation engine matches skills with internship requirements

---

## 📈 Performance

- **Extraction Speed**: ~0.15 seconds per resume
- **Batch Processing**: ~6-8 resumes per second
- **Skill Detection**: 100+ skills with pattern matching
- **Accuracy**: High precision for common skills and standard resume formats
- **Scalability**: Can process thousands of resumes in minutes

---

## 🎨 User Interface

The frontend features:
- ✨ **Animated gradient background** with floating orbs
- 🎯 **Drag-and-drop file upload** with visual feedback
- 💎 **Skill tags** with smooth animations
- 📊 **Internship cards** with hover effects
- 📱 **Responsive design** for mobile and desktop
- 🌈 **Modern color scheme** with glassmorphism effects

---

## 🔒 Privacy & Security

- ✅ Uploaded resumes are processed in temporary files
- ✅ Files are automatically deleted after processing
- ✅ No data is stored permanently without user consent
- ✅ CORS enabled for secure frontend-backend communication
- ✅ No external API calls for resume processing

---

## 📚 Usage Guide

### For End Users:

1. **Open the application** in your browser (`http://localhost:5173`)
2. **Click "Choose File"** or drag-and-drop your resume PDF
3. **Click "Start Analysis"** to extract skills
4. **View extracted skills** displayed as colorful tags
5. **Click "Get Recommendations"** to see matching internships
6. **Browse internship cards** and click "View Details & Apply"

### For Developers:

1. **Process dataset**: Run `python process_all_resumes.py` to generate Excel files
2. **Customize skills**: Edit `backend/extractor.py` to add/remove skills
3. **Add internships**: Update `backend/data/dummy_internship_recommendations.json`
4. **Modify UI**: Edit `frontend/src/App.jsx` for frontend changes
5. **API testing**: Use the `/health` endpoint to verify backend is running

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is already in use
# On Windows:
netstat -ano | findstr :8000

# On Linux/Mac:
lsof -i :8000

# Kill the process and restart
```

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check `frontend/vite.config.js` proxy settings
- Verify CORS is enabled in `backend/app.py`

### PDF extraction fails
- Ensure the PDF is not password-protected
- Check if PyPDF2 is installed: `pip install PyPDF2`
- Try with a different PDF file

### No skills detected
- Ensure the resume contains text (not just images)
- Check if skills are in the predefined list in `backend/extractor.py`
- Try with a standard resume format

---

## 🤝 Contributing

This is an academic project (FYP-65). For improvements or bug fixes:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

Academic project - All rights reserved

---

## 👥 Authors

- **Project**: Final Year Project (FYP-65)
- **Institution**: [Your University Name]
- **Year**: 2026

---

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- Chakra UI for beautiful React components
- PyPDF2 for PDF processing capabilities
- The open-source community for inspiration

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review the code comments in `backend/app.py` and `backend/extractor.py`
- Test with the sample PDFs in `DATA/data/`

---

**Last Updated**: February 2026

**Status**: ✅ Production Ready

**Version**: 1.0.0
