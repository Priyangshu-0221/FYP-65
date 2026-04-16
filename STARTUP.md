# 🚀 Full Stack Startup Guide

## Prerequisites

- **Node.js** 18.17+ (for frontend)
- **Python** 3.8+ (for backend)

---

## Quick Start (All-in-One)

### Windows PowerShell

```powershell
# Navigate to project root
cd C:\Users\priya\Downloads\FYP-65

# Run the startup script
.\start-all.ps1
```

---

## Manual Startup (Two Terminals)

### Terminal 1: Backend API Server

```bash
# Navigate to project root
cd C:\Users\priya\Downloads\FYP-65

# Install dependencies (first time only)
pip install -r requirements.txt

# Start backend on port 8000
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

✅ **Backend Ready at:** `http://localhost:8000`

#### Test Backend Health:

```bash
curl http://localhost:8000/health
```

---

### Terminal 2: Frontend Dev Server

```bash
# Navigate to frontend directory
cd C:\Users\priya\Downloads\FYP-65\frontend-next

# Install dependencies (first time only)
npm install

# Start frontend on port 3000
npm run dev
```

**Expected Output:**

```
▲ Next.js 16.2.3 (Turbopack)
- Local:   http://localhost:3000
- Network: http://192.168.x.x:3000
✓ Ready in 1234ms
```

✅ **Frontend Ready at:** `http://localhost:3000`

---

## 🌐 Complete Stack Status

Once both servers are running:

| Service         | URL                          | Status     |
| --------------- | ---------------------------- | ---------- |
| **Frontend**    | http://localhost:3000        | ✅ Running |
| **Backend API** | http://localhost:8000        | ✅ Running |
| **API Health**  | http://localhost:8000/health | ✅ OK      |

---

## 📋 Available API Endpoints

### Health Check

```bash
GET http://localhost:8000/health
```

### Upload Resume (Extract Skills)

```bash
POST http://localhost:8000/upload
Content-Type: multipart/form-data
Body: file (PDF)
```

**Response:**

```json
{
  "success": true,
  "skills": ["python", "react", "sql", ...],
  "cgpa": 8.5,
  "percentage": 85,
  "education": "...",
  "experience": "..."
}
```

### Get Recommendations

```bash
POST http://localhost:8000/recommend
Content-Type: application/json
Body: {
  "skills": ["python", "react", "sql"],
  "top_k": 6,
  "marks": { "cgpa": 8.5 }
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "id": "1",
      "title": "Software Developer Intern",
      "company": "Tech Company",
      "location": "Remote",
      "category": "Software",
      "skills": ["python", "java", "sql"],
      "description": "...",
      "apply_link": "#"
    },
    ...
  ]
}
```

---

## 🧪 Testing the Full Workflow

1. **Open Frontend:** http://localhost:3000
2. **Upload a Resume:** Use the upload component on dashboard
3. **View Extracted Skills:** Skills appear after successful upload
4. **Add Academic Info:** Enter marks (optional)
5. **Get Recommendations:** Click "Get Recommendations" button
6. **View Results:** Personalized internship matches appear below

---

## 🛠️ Development Commands

### Frontend

```bash
cd frontend-next

# Development mode with hot reload
npm run dev

# Build for production
npm run build

# Start production server
npm run start

# Lint code
npm run lint
```

### Backend

```bash
# Run with auto-reload on file changes
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Run without reload
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Run from __main__.py
python -m backend
```

---

## 📁 Project Structure

```
FYP-65/
├── frontend-next/          # Next.js 16 React App
│   ├── src/
│   │   ├── app/            # Pages & routes
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom hooks
│   │   ├── services/       # API calls
│   │   └── types/          # TypeScript types
│   ├── package.json
│   └── .env.local
│
├── backend/                # FastAPI Python App
│   ├── app.py              # Main API
│   ├── extractor.py        # Resume processing
│   ├── __main__.py         # Entry point
│   └── settings.py         # Config
│
├── requirements.txt        # Python dependencies
└── start-all.ps1          # Startup script

```

---

## ⚠️ Troubleshooting

### Port Already in Use

```bash
# Kill process using port 8000 (Backend)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Kill process using port 3000 (Frontend)
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

### Module Not Found (Python)

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### npm Dependencies Issue

```bash
# Clear and reinstall
rm -r node_modules package-lock.json
npm install
```

---

## 🎯 Features

✅ **Resume Upload & Parsing** - Extract skills from PDF  
✅ **AI-Powered Matching** - Get personalized internship recommendations  
✅ **Academic Integration** - Factor in CGPA/percentage  
✅ **Modern UI** - Fully responsive design  
✅ **Real-time Updates** - Hot reload during development

---

## 📞 Support

For issues or questions, check the logs in each terminal for detailed error messages.
