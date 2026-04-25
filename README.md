<div align="center">

# 🚀 AI Powered Career Guidance System

> **Intelligent Resume Analysis & Semantic Internship Matching Platform**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img alt="Next.js" src="https://img.shields.io/badge/Next.js_16-000000?style=for-the-badge&logo=next.js&logoColor=white" />
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
  <img alt="Tailwind CSS" src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" />
  <img alt="AI/ML" src="https://img.shields.io/badge/AI%2FML-Sentence%20Transformers-FF6B35?style=for-the-badge" />
</p>

🎓 _Bachelor of Technology (CSE) Final Year Project_  
📍 _Sister Nivedita University, 2025_

[📘 Backend Docs](./backend/README.md) · [🎨 Frontend Docs](./frontend-next/README.md) · [🚀 Quick Start](#-quick-start)

</div>

---

## 📖 About

**AI Powered Career Guidance System** is a comprehensive full-stack platform that intelligently analyzes PDF resumes, extracts competencies, and provides semantic-based internship recommendations using state-of-the-art AI techniques. The system features a modern React/Next.js frontend with a powerful FastAPI backend powered by Sentence Transformers for accurate skill-to-opportunity matching.

---

## ✨ Core Features

### 🧠 Backend: Resume Processing & AI Recommendations

| Feature                  | Details                                        | Technology            |
| ------------------------ | ---------------------------------------------- | --------------------- |
| **📄 PDF Extraction**    | Extracts text from PDF resumes                 | PyPDF2                |
| **🎯 Skill Recognition** | Identifies 500+ technical & soft skills        | NLP Pattern Matching  |
| **🤖 Semantic Matching** | AI-powered internship matching                 | Sentence Transformers |
| **📊 Data Analysis**     | Extract education, experience, CGPA/percentage | PyPDF2 + Regex        |
| **⚡ Async Processing**  | Non-blocking file uploads                      | FastAPI async         |
| **📈 Scoring System**    | Relevance-based ranking                        | Cosine Similarity     |

### 🎨 Frontend: Modern Interactive Dashboard

| Feature                   | Details                                               | Technology           |
| ------------------------- | ----------------------------------------------------- | -------------------- |
| **🏠 Welcome Page**       | Academic showcase with project details                | Next.js App Router   |
| **📋 4-Step Workflow**    | Guided resume → skills → profile → recommendations    | React Components     |
| **💼 Results Dashboard**  | Internship cards, skill analysis, gap recommendations | Tailwind CSS v4      |
| **🎯 Real-time Progress** | Visual progress tracking with smooth animations       | React State          |
| **📱 Responsive Design**  | Mobile-first, works on all devices                    | Tailwind Responsive  |
| **♿ Accessibility**      | WCAG AA compliant, keyboard navigation                | Semantic HTML + ARIA |

---

## 🛠️ Technology Stack

### Backend (Python/FastAPI)

<div align="center">

| Layer               | Tech                        | Purpose                |
| ------------------- | --------------------------- | ---------------------- |
| **Framework**       | FastAPI 0.100+              | Async web framework    |
| **Language**        | Python 3.10+                | ML & data science      |
| **AI/ML**           | Sentence Transformers       | Semantic embeddings    |
| **PDF Processing**  | PyPDF2                      | Text extraction        |
| **Data Processing** | Pandas, NumPy, Scikit-learn | Numerical computing    |
| **Server**          | Uvicorn                     | ASGI server            |
| **Config**          | Pydantic Settings           | Environment management |

</div>

### Frontend (React/Next.js)

<div align="center">

| Layer             | Tech            | Purpose                |
| ----------------- | --------------- | ---------------------- |
| **Framework**     | Next.js 16.2.3  | React meta-framework   |
| **Language**      | TypeScript 5.x  | Type-safe development  |
| **Styling**       | Tailwind CSS v4 | Utility-first CSS      |
| **UI Components** | Shadcn-inspired | Reusable primitives    |
| **Icons**         | Lucide React    | Consistent iconography |
| **Build Tool**    | Turbopack       | Fast bundling          |
| **HTTP**          | Fetch API       | Backend communication  |

</div>

---

## 📁 Project Architecture

```
FYP-65/                                 # Project Root
│
├── 🧠 backend/                         # FastAPI Backend (Port 8080)
│   ├── app.py                          # FastAPI application & routes
│   ├── extractor.py                    # Resume PDF extraction
│   ├── recommendation_engine.py        # AI/ML matching engine
│   ├── internship_catalog.py           # Database queries
│   ├── schemas.py                      # Data validation
│   ├── settings.py                     # Configuration
│   ├── data/
│   │   └── dummy_internship_recommendations.json
│   ├── __main__.py                     # Entry point
│   └── README.md                       # Backend documentation
│
├── 🎨 frontend-next/                   # Next.js Frontend (Port 3000)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                # Welcome page
│   │   │   ├── layout.tsx              # Root layout
│   │   │   └── globals.css             # Theme & animations
│   │   ├── components/
│   │   │   ├── features/
│   │   │   │   ├── DashboardWorkspace.tsx
│   │   │   │   ├── UploadSection.tsx
│   │   │   │   ├── SkillsList.tsx
│   │   │   │   ├── AcademicMarksSection.tsx
│   │   │   │   ├── RecommendationsGrid.tsx
│   │   │   │   └── SkillSuggestions.tsx
│   │   │   ├── layout/
│   │   │   ├── overlays/
│   │   │   └── ui/
│   │   ├── hooks/
│   │   ├── services/
│   │   │   └── api.ts                  # Backend client
│   │   ├── types/
│   │   └── lib/
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   └── README.md                       # Frontend documentation
│
├── 📊 DATA/                            # Resume Dataset (24 categories)
│   ├── data/
│   │   ├── ACCOUNTANT/
│   │   ├── ENGINEERING/
│   │   ├── INFORMATION-TECHNOLOGY/
│   │   └── ... (21 more categories)
│   └── processed/
│       ├── raw_data.xlsx
│       ├── cleaned_data.xlsx
│       └── tokenized_data.xlsx
│
├── 🔧 scripts/
│   ├── process_all_resumes.py          # Batch processing
│   └── evaluate_model.py
│
├── requirements.txt                    # Python dependencies
├── start-all.sh                        # Start both services (Unix)
├── start-all.ps1                       # Start both services (Windows)
├── STARTUP.md                          # Detailed startup guide
└── README.md                           # This file
```

---

## 🚀 Quick Start

### 📋 Prerequisites

```bash
✓ Python 3.10+         (Backend runtime)
✓ Node.js 18+          (Frontend runtime)
✓ pip & npm            (Package managers)
✓ 2GB+ RAM             (For AI models)
✓ Internet connection  (First-time setup)
```

### 💻 Installation & Setup

**1️⃣ Clone the repository:**

```bash
cd FYP-65
```

**2️⃣ Create Python virtual environment:**

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

**3️⃣ Install Python dependencies:**

```bash
pip install -r requirements.txt
```

> 📌 First-time installation downloads AI models (~300-400MB). This may take 2-3 minutes.

**4️⃣ Install frontend dependencies:**

```bash
cd frontend-next
npm install
cd ..
```

### ▶️ Running the Application

**Option 1: Automated (Windows PowerShell)**

```powershell
.\start-all.ps1
```

**Option 2: Automated (Linux/macOS)**

```bash
bash start-all.sh
```

**Option 3: Manual (Two Terminals)**

**Terminal 1 - Backend:**

```bash
# Make sure venv is activated
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080
```

**Terminal 2 - Frontend:**

```bash
cd frontend-next
npm run dev
```

### 🌐 Access the Application

- **Frontend:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8080](http://localhost:8080)
- **API Documentation:** [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🔌 API Endpoints

### Backend API (Port 8080)

| Endpoint     | Method | Purpose                        | Response                            |
| ------------ | ------ | ------------------------------ | ----------------------------------- |
| `/health`    | GET    | Server availability check      | `{ "status": "ok" }`                |
| `/upload`    | POST   | Upload PDF & extract skills    | Skills, education, experience, CGPA |
| `/recommend` | POST   | Get internship recommendations | Matched internships + skill gaps    |

**Example API Usage:**

```bash
# Check backend health
curl http://localhost:8080/health

# Upload resume
curl -X POST -F "file=@resume.pdf" http://localhost:8080/upload

# Get recommendations
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"skills": ["python", "react"], "top_k": 6}'
```

---

## 📊 Data Processing

### Batch Resume Processing

Process all resumes and generate Excel files:

```bash
python scripts/process_all_resumes.py
```

**Generated Files (in `DATA/processed/`):**

1. **raw_data.xlsx** – Original text + extracted metadata
2. **cleaned_data.xlsx** – Lowercase, no special characters
3. **tokenized_data.xlsx** – Tokenized text for NLP

**Processing Speed:** ~6-8 resumes/second

---

## 📂 Detailed Documentation

### Frontend Documentation

For detailed frontend documentation including component structure, design system, and development guide:

📘 **[→ Frontend-Next README](./frontend-next/README.md)**

Key sections:

- 🎨 Design system & theme configuration
- 🏗️ Component architecture
- 📝 Available scripts & build commands
- 🔗 Backend integration guide

### Backend Documentation

For detailed backend documentation including API specs, recommendation engine, and deployment guide:

🧠 **[→ Backend README](./backend/README.md)**

Key sections:

- 🤖 Recommendation engine details
- 📄 Resume extraction pipeline
- 🔌 Complete API documentation
- ⚙️ Configuration & environment setup

---

## 🎯 Workflow

### End-to-End User Journey

```
1. User Opens App (http://localhost:3000)
   ↓
2. Welcome Page - Project Overview & Quick Stats
   ↓
3. Upload Resume PDF - Drag & drop interface
   ↓
4. Backend Processing:
   - PDF text extraction
   - Skill recognition (500+ skills)
   - Education/experience parsing
   - CGPA/percentage detection
   ↓
5. Extracted Skills Display - Color-coded badges
   ↓
6. Academic Profile Entry - CGPA, percentage input
   ↓
7. Generate Recommendations - AI matching process
   ↓
8. Results View:
   - Top internships (sorted by relevance)
   - Skill gap analysis
   - Direct apply links
```

---

## 🧪 Development Workflow

### Frontend Development

```bash
cd frontend-next

# Development with hot reload
npm run dev

# Build for production
npm run build
npm run start

# Linting
npm run lint
```

### Backend Development

```bash
# Development with auto-reload
python -m uvicorn backend.app:app --reload --port 8080

# Debug mode with verbose logging
python -m uvicorn backend.app:app --reload --log-level debug

# Production mode
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080
```

---

## 🛡️ Features Highlight

### Resume Processing

✅ **Supports:** PDF files  
✅ **Speed:** ~0.15 seconds per resume  
✅ **Accuracy:** High precision for standard formats  
✅ **Extraction:** Skills, education, experience, CGPA, percentage

### Internship Database

✅ **25+ Opportunities** – Pre-loaded dataset  
✅ **Multiple Sectors** – Tech, Finance, HR, Design, Healthcare, Engineering  
✅ **Global Locations** – Remote, India, UK, Singapore, USA  
✅ **Rich Metadata** – Company, location, skills, descriptions

### AI Recommendation Engine

✅ **Semantic Matching** – Sentence Transformers embeddings  
✅ **Flexible Models** – Switch between 3 industry-standard models  
✅ **Scoring System** – Cosine similarity-based ranking  
✅ **Skill Gap Analysis** – Identify missing competencies

### User Experience

✅ **Modern Dark Theme** – Black & gold aesthetic  
✅ **Responsive Design** – Mobile to desktop  
✅ **Real-time Progress** – Visual step tracking  
✅ **Accessible** – WCAG AA compliant

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in root directory:

```env
# Backend Configuration
INTERNSHIP_APP_BACKEND_HOST=0.0.0.0
INTERNSHIP_APP_BACKEND_PORT=8080
INTERNSHIP_APP_LOG_LEVEL=info

# Model Configuration
INTERNSHIP_APP_MODEL_NAME=BAAI/bge-small-en-v1.5

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8080
```

### Recommendation Models

Available models in `backend/recommendation_engine.py`:

| Model         | Speed  | Accuracy   | Best For               |
| ------------- | ------ | ---------- | ---------------------- |
| **minilm**    | ⚡⚡⚡ | ⭐⭐⭐     | Low latency            |
| **bge-small** | ⚡⚡   | ⭐⭐⭐⭐⭐ | Balanced (recommended) |
| **mpnet**     | ⚡     | ⭐⭐⭐⭐   | Maximum accuracy       |

---

## 🚀 Deployment

### Docker Deployment (Optional)

Build and run using Docker:

```bash
# Build images
docker build -t fyp-backend ./backend
docker build -t fyp-frontend ./frontend-next

# Run containers
docker run -p 8080:8080 fyp-backend
docker run -p 3000:3000 fyp-frontend
```

---

## 🐛 Troubleshooting

### Backend Issues

| Issue                    | Solution                                     |
| ------------------------ | -------------------------------------------- |
| **Port 8080 in use**     | Change port: `--port 8081`                   |
| **Module import errors** | Reinstall: `pip install -r requirements.txt` |
| **PDF extraction fails** | Verify PDF is valid, not password-protected  |
| **Slow recommendations** | Switch to `minilm` model in settings         |
| **No skills detected**   | Ensure PDF contains text (not image-based)   |

### Frontend Issues

| Issue                        | Solution                                          |
| ---------------------------- | ------------------------------------------------- |
| **Can't connect to backend** | Ensure backend running on 8080                    |
| **Port 3000 in use**         | Use different port: `npm run dev -- --port 3001`  |
| **Dependencies error**       | Clear cache: `rm -rf node_modules && npm install` |
| **Styling issues**           | Rebuild: `npm run build`                          |

### Common Commands

```bash
# Check if ports are in use
# Windows
netstat -ano | findstr :8080
netstat -ano | findstr :3000

# Linux/macOS
lsof -i :8080
lsof -i :3000

# Kill a process
# Windows
taskkill /PID <PID> /F

# Linux/macOS
kill -9 <PID>
```

---

## 📊 Performance Metrics

| Metric                  | Value                            |
| ----------------------- | -------------------------------- |
| **Resume Extraction**   | ~0.15 sec/file                   |
| **Batch Processing**    | ~6-8 resumes/sec                 |
| **Skill Detection**     | 500+ skills                      |
| **Recommendation Time** | <1 second                        |
| **Accuracy**            | High (>90% for standard formats) |

---

## ✅ Testing Checklist

- [ ] Backend starts without errors on port 8080
- [ ] Frontend starts without errors on port 3000
- [ ] Health endpoint responds: `GET http://localhost:8080/health`
- [ ] Can upload PDF and extract skills: `POST /upload`
- [ ] Can get recommendations: `POST /recommend`
- [ ] Frontend loads at `http://localhost:3000`
- [ ] Can upload resume through UI
- [ ] Recommendations generate within 1 second
- [ ] Responsive on mobile devices

---

## 🎓 Project Information

| Detail           | Information                             |
| ---------------- | --------------------------------------- |
| **Project Name** | AI Powered Career Guidance System       |
| **Project Type** | Final Year Project (FYP-65)             |
| **University**   | Sister Nivedita University (SNU)        |
| **Program**      | Bachelor of Technology (BTech) - CSE    |
| **Supervisor**   | Dr. Sayani Mondal (Assistant Professor) |
| **Submission**   | November 25, 2025                       |
| **Session**      | Academic Year 2025                      |
| **Status**       | ✅ Production Ready                     |

### Development Team

| Name              | Registration | Email                        |
| ----------------- | ------------ | ---------------------------- |
| Priyangshu Mondal | 220100663543 | mondalpriyangshu@gmail.com   |
| Abhijit Biswas    | 220100017663 | abhijit.biswas1024@gmail.com |
| Kunal Roy         | 220100185465 | royku321@gmail.com           |
| Rupam Haldar      | 220100408950 | prabirhaldar68@gmail.com     |

---

## 📚 Key Technologies

### Core Stack

- **Python 3.10+** – Backend runtime
- **FastAPI 0.100+** – Web framework
- **Next.js 16.2.3** – Frontend framework
- **TypeScript 5.x** – Type-safe development
- **Tailwind CSS v4** – Styling

### AI & ML

- **Sentence Transformers** – Semantic matching
- **PyPDF2** – PDF processing
- **NumPy** – Numerical computing
- **Scikit-learn** – ML algorithms

### DevOps

- **Uvicorn** – ASGI server
- **Turbopack** – Build tool
- **CORS Middleware** – Cross-origin handling

---

## 📖 Documentation

- 📘 **[Backend Documentation](./backend/README.md)** – API specs, architecture, deployment
- 🎨 **[Frontend Documentation](./frontend-next/README.md)** – Components, design system, development
- 🚀 **[Startup Guide](./STARTUP.md)** – Detailed setup instructions

---

## 🔒 Security & Privacy

✅ **Privacy:**

- Uploaded PDFs processed in temporary files
- Files automatically deleted after processing
- No permanent data storage without consent

✅ **Security:**

- PDF file validation (only .pdf accepted)
- File size limits enforced
- CORS enabled with origin validation
- Error messages don't leak sensitive info
- Async processing prevents blocking attacks

---

## 🤝 Contributing

For academic purposes, contributions welcome:

1. Fork repository
2. Create feature branch: `git checkout -b feature/improvement`
3. Make changes with tests
4. Submit pull request with description
5. Link to relevant issues

---

## 📄 License

This project is created as part of an academic Final Year Project and is intended for educational purposes only.

**All rights reserved © 2025 Sister Nivedita University**

---

## 🙏 Acknowledgments

- **FastAPI Team** – Excellent async web framework
- **Sentence Transformers** – State-of-the-art embeddings
- **Next.js Team** – Modern React framework
- **Tailwind CSS** – Utility-first CSS framework
- **Open Source Community** – For inspiration & tools

---

<p align="center">

### 🎉 Thank You!

This project represents months of research, development, and testing.  
Your feedback and contributions help improve the system!

**Built with ❤️ using Python, FastAPI, React, and Next.js**

[📞 Contact us](mailto:mondalpriyangshu@gmail.com) · [📊 View Backend Docs](./backend/README.md) · [🎨 View Frontend Docs](./frontend-next/README.md)

**Last Updated:** April 19, 2026  
**Version:** 1.0.0  
**Status:** ✅ Production Ready

</p>
