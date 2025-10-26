# FYP-65 Internship Recommender

This project transforms resumes into structured insights, trains a classifier on resume categories, and exposes a web-based internship recommender that matches extracted skills to curated opportunities.

---

## Repository Structure

| Path | Description |
| --- | --- |
| `resume_feature_engineering.py` | Enriches CSV resume datasets with derived features (skills, education, experience). |
| `pdf_resume_feature_engineering.py` | Extracts features from PDF resumes and exports structured summaries. |
| `resume_classifier.py` | Trains TF-IDF based classifier (Naive Bayes or Logistic Regression) on resume categories. |
| `web_backendbackend | FastAPI service exposing resume upload and internship recommendation endpoints. |
| `frontend/` | React + Vite application (Chakra UI) for uploading resumes and viewing recommendations. |
| `setup_all_dependencies.py` | Automation script to install Python and Node dependencies in one step. |
| `requirements.txt` | Python dependency lockfile. |
| `DATA/` | Placeholder for resume datasets (CSV or folder of categorized PDFs). |

---

## Quick Start

1. **Clone/Download** this repository and ensure Python 3.10+ and Node.js 18+ are available on your machine.
2. **Place a resume dataset** at `DATA/resumes.csv` (columns: `Category`, `Resume`) or organise categorized PDFs under `DATA/`.
3. **Install all dependencies** with a single command:

   ```bash
   python setup_all_dependencies.py
   ```

4. **Run the backend** (from project root):

   ```bash
   uvicorn backend.fastapi_app:app --reload
   ```

5. **Run the frontend** (in a separate terminal):

   ```bash
   cd frontend
   npm run dev
   ```

6. Open the presented Vite URL (default `http://localhost:5173`). Upload a resume and request recommendations.

---

## Detailed Setup

### 1. Python Environment

*Ensure you are inside a virtual environment (recommended).*  
Dependencies are defined in `requirements.txt` and installed automatically via the setup script. Manual installation:

```bash
pip install -r requirements.txt
```

The backend relies on NLTK corpora (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`). These download automatically on first run, but you can prefetch them:

```python
import nltk
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource)
```

### 2. Node Environment

The React UI lives in `frontend/`. After running `setup_all_dependencies.py`, node modules are installed. Manual installation:

```bash
cd frontend
npm install
```

Vite configuration (`vite.config.js`) proxies `/api` requests to the FastAPI backend at `http://127.0.0.1:8000`.

---

## Running the Components

### Backend (FastAPI)

```bash
uvicorn backend.fastapi_app:app --reload
```

Key endpoints:

| Endpoint | Method | Description |
| --- | --- | --- |
| `/upload` | POST | Accepts PDF/TXT resumes, extracts skills, and returns predicted category (if model trained). |
| `/recommend` | POST | Returns ranked internships based on supplied skills. |

Environment configuration is handled via `backend/backend.py`. Override defaults with environment variables (prefix `INTERNSHIP_APP_`), e.g.:

```bash
set INTERNSHIP_APP_RESUME_DATASET_PATH=DATA/resumes.csv
set INTERNSHIP_APP_INTERNSHIP_CATALOG_PATH=custom_catalog.json
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

