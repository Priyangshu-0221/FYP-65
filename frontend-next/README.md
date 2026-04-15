<div align="center">
  <img src="public/snu-logo.png" alt="Sister Nivedita University Logo" width="300"/>

# AI Powered Career Guidance System

**Resume Analyser and Internship Recommendation Engine**

🎓 _Bachelor of Technology (CSE) Final Year Project_

</div>

<br />

The **AI Powered Career Guidance System** frontend is an interactive, modern web application built with Next.js 16 (App Router) and Tailwind CSS v4. It provides university students with an intuitive, 4-step workflow to analyze their resumes, extract relevant skills in real-time, integrate academic performance metrics, and generate AI-driven internship recommendations.

---

## ✨ Features and Capabilities

- **Responsive Academic Welcome Page:** A sleek, fully responsive title page (adaptable to print as a formal report cover) displaying essential project tracking information and project contributors.
- **Intuitive 4-Step Dashboard Workflow:**
  1.  **Resume Upload (Step 1):** Fast, reliable drag-and-drop resume upload system natively integrated with our backend parser.
  2.  **Skill View (Step 2):** Real-time display of AI-extracted skills from the parsed resume.
  3.  **Academic Input (Step 3):** Supplemental input for academic metrics to refine match accuracy.
  4.  **Internship Matching (Step 4):** A visually engaging, grid-based card layout delivering personalized, highly relevant internship recommendations.
- **Modern Design System:** Completely bespoke university-styled theme relying on deep blues (`#1d3b72`), accessible fluid typographies (`Inter`), card-surface shadows, and `lucide-react` iconography for professional visual feedback.

---

## 🛠️ Technology Stack

| Technology              | Description                                                           |
| :---------------------- | :-------------------------------------------------------------------- |
| **Framework**           | [Next.js 16.2.3](https://nextjs.org/) (App Router, Turbopack)         |
| **Styling**             | [Tailwind CSS v4](https://tailwindcss.com/) & Custom CSS Variables    |
| **Icons**               | [Lucide React](https://lucide.dev/)                                   |
| **UI Building Blocks**  | Radix / Shadcn-inspired custom primitives (`Button`, `Card`, `Badge`) |
| **Language**            | TypeScript                                                            |
| **Toast Notifications** | Sonner                                                                |

---

## 📂 Frontend Directory Structure

The project structure strictly adheres to the Next.js App Router guidelines and clean feature-based architecture.

```text
C:\Users\priya\Downloads\FYP-65\frontend-next\
├── public/                 # Static assets
│   └── snu-logo.png        # University crest banner used on Welcome Title Page
├── src/                    # Main source code
│   ├── app/                # App Router and Routing
│   │   ├── globals.css     # Global theme configuration, animations & CSS variables
│   │   ├── layout.tsx      # Root application shell, Inter Font, and global layouts
│   │   ├── page.tsx        # Project Welcome / Formal Academic Title Page
│   │   └── dashboard/      # Main workflow app entry point
│   │       └── page.tsx    # Responsive 4-Step Workflow View
│   ├── components/         # Highly reusable frontend UI fragments
│   │   ├── features/       # Workflow sections (UploadSection, SkillsList, etc.)
│   │   ├── layout/         # Core layout wrappers (Header, Sidebar)
│   │   └── ui/             # Baseline UI primitives (Card, Button, Input, Badge)
│   ├── hooks/              # Custom React Hooks
│   │   ├── useRecommendations.ts # API link to Recommendation Engine
│   │   └── useResumeUpload.ts    # Resume transmission state & validation handling
│   ├── services/           # Backend API interaction layers
│   │   └── api.ts          # Axios / fetch calls & interface definitions
│   └── types/              # Global TypeScript interfaces
│       ├── api.ts          # Server response typing
│       └── app.ts          # Application-level structural typings
├── package.json            # NPM dependencies and scripts
└── README.md               # You are here!
```

---

## 🎨 Theme & Design Tokens

The application leverages global CSS variables set inside `src/app/globals.css`.

- **Primary Background:** `--bg: #f4f7fb;`
- **Card Surfaces:** `--surface: #ffffff;`
- **Accent (University Blue):** `--accent: #1d3b72;`
- **Typography:** Next.js `next/font/google` implemented Inter sans-serif for universally clean spacing.

### Advanced CSS Capabilities

- Built-in Scrollbar optimizations (`::-webkit-scrollbar`).
- Embedded print logic (`@media print`) enabling seamless PDF generation of the academic root page directly from the browser window.
- Fully dynamic and stacking flexbox layout suitable natively for Android, iOS, tablets, and wide-desktop monitoring.

---

## 🚀 Getting Started

### Prerequisites

- Node.js (`v18.17.0` or higher)

### Setup & Installation

**1. Navigate to the frontend directory:**

```bash
cd frontend-next
```

**2. Install dependencies:**

```bash
npm install
```

**3. Start the development server:**

```bash
npm run dev
# or with Turbopack (if compatible)
npm run dev --turbo
```

**4. Build for Production:**

```bash
npm run build
npm run start
```

_Note: A successful build guarantees that the application passes all TypeScript strict typings and CSS bundle validations!_

---

## 👥 Meet the Team

**Guided By:** Dr. Sayani Mondal _(Assistant Professor)_

**Developed By:**

- **Priyangshu Mondal** (Reg: `220100663543`, `mondalpriyangshu@gmail.com`)
- **Abhijit Biswas** (Reg: `220100017663`, `abhijit.biswas1024@gmail.com`)
- **Kunal Roy** (Reg: `220100185465`, `royku321@gmail.com`)
- **Rupam Haldar** (Reg: `220100408950`, `prabirhaldar68@gmail.com`)

_Submitted: November 2025_
_Academic Session: 2025_

---

<p align="center">Made with ❤️ using Next.js & Tailwind CSS.</p>
