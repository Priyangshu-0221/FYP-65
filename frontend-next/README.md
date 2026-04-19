<div align="center">

# 🚀 AI Powered Career Guidance System - Frontend

> **Resume Analyser and Internship Recommendation Engine**

<p>
  <img alt="Next.js" src="https://img.shields.io/badge/Next.js_16-000000?style=for-the-badge&logo=next.js&logoColor=white" />
  <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
  <img alt="Tailwind CSS" src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" />
  <img alt="React" src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
</p>

🎓 _Bachelor of Technology (CSE) Final Year Project_  
📍 _Sister Nivedita University, 2025_

</div>

---

## 📖 About

The **AI Powered Career Guidance System Frontend** is a cutting-edge, high-performance web application built with **Next.js 16** and **Tailwind CSS v4**. It delivers a sleek dark-themed interface that guides students through an intelligent 4-step workflow to analyze resumes, extract competencies, review academic profiles, and discover AI-driven internship opportunities perfectly matched to their skills and qualifications.

---

## ✨ Key Features

### 🏠 Academic Welcome Page

A professional landing page featuring:

- 📋 **Project Showcase** – Complete project details, supervisors, and team information
- 🎓 **Institutional Branding** – Sister Nivedita University logo and academic context
- ⚡ **Quick Access** – One-click jump to the recommendation engine
- 🖨️ **Print-Friendly** – Optimized for formal academic documentation

### 🎯 Interactive 4-Step Recommendation Engine

The core workflow with **real-time progress tracking**:

| Step           | Feature           | Details                                                         |
| -------------- | ----------------- | --------------------------------------------------------------- |
| **1️⃣ Upload**  | Resume File Input | Drag-and-drop interface, real-time feedback, FastAPI processing |
| **2️⃣ Extract** | Skill Analysis    | AI-extracted competencies, color-coded badges, live counter     |
| **3️⃣ Profile** | Academic Metrics  | CGPA/GPA input, percentage tracking, integrated algorithms      |
| **4️⃣ Match**   | Recommendations   | One-click generation, AI matching, visual processing feedback   |

### 📊 Intelligent Results Dashboard

**Right-Side Processing Overview** featuring:

- 🎯 **Sticky Status Panel** – Always-visible tracking:
  - Current resume file name
  - Number of extracted skills
  - Recommended internship count

- 💼 **Recommended Internships Grid** – Primary results with:
  - Title, company, location
  - Match percentage & relevance score
  - Required skills & descriptions
  - Direct apply links

- 🔍 **Skill Gap Analysis** – Secondary recommendations:
  - Missing skills for target roles
  - Prioritized upskilling paths
  - Competency gap insights

---

## 🛠️ Tech Stack

<div align="center">

| Layer                | Technology      | Purpose                                    |
| -------------------- | --------------- | ------------------------------------------ |
| **🎯 Framework**     | Next.js 16.2.3  | React meta-framework & SSR                 |
| **📝 Language**      | TypeScript 5.x  | Type-safe development                      |
| **🎨 Styling**       | Tailwind CSS v4 | Utility-first CSS                          |
| **🎭 Icons**         | Lucide React    | Consistent iconography                     |
| **🧩 Components**    | Shadcn-inspired | UI primitives (Button, Card, Input, Badge) |
| **🔔 Notifications** | React-Toastify  | Toast alerts & feedback                    |
| **🔗 HTTP**          | Fetch API       | Backend communication                      |
| **⚡ Build Tool**    | Turbopack       | Fast development builds                    |

</div>

---

## 🛠️ Technology Stack

| Layer             | Technology                                                  | Purpose                                  |
| :---------------- | :---------------------------------------------------------- | :--------------------------------------- |
| **Framework**     | [Next.js 16.2.3](https://nextjs.org/) (App Router)          | React meta-framework, SSR, routing       |
| **Language**      | TypeScript 5.x                                              | Type-safe development                    |
| **Styling**       | [Tailwind CSS v4](https://tailwindcss.com/)                 | Utility-first CSS framework              |
| **Icons**         | [Lucide React](https://lucide.dev/)                         | Beautiful, consistent iconography        |
| **UI Primitives** | Custom Shadcn-inspired components                           | Button, Card, Input, Badge, Dialog, Tabs |
| **Notifications** | [React-Toastify](https://fkhadra.github.io/react-toastify/) | Toast alerts and feedback                |
| **HTTP Client**   | Fetch API (Native)                                          | Backend API communication                |
| **Build Tool**    | Turbopack (Next.js integrated)                              | Fast development builds                  |

---

## 📂 Project Structure

```
frontend-next/
│
├── 📁 public/
│   └── snu-logo.png              # University Logo
│
├── 📁 src/
│   ├── 📁 app/
│   │   ├── globals.css           # Theme + Animations
│   │   ├── layout.tsx            # Root Layout
│   │   ├── page.tsx              # Welcome Page
│   │   └── 📁 dashboard/
│   │       └── page.tsx          # Legacy Route
│   │
│   ├── 📁 components/
│   │   ├── 📁 features/
│   │   │   ├── DashboardWorkspace.tsx      # Main Workflow
│   │   │   ├── UploadSection.tsx           # File Upload
│   │   │   ├── SkillsList.tsx              # Skills Display
│   │   │   ├── AcademicMarksSection.tsx    # CGPA Input
│   │   │   ├── RecommendationsGrid.tsx     # Results Grid
│   │   │   └── SkillSuggestions.tsx        # Gap Analysis
│   │   │
│   │   ├── 📁 layout/
│   │   │   ├── Header.tsx        # Navigation
│   │   │   └── Sidebar.tsx       # Side Nav
│   │   │
│   │   ├── 📁 overlays/
│   │   │   ├── SearchingBooksLoader.tsx    # Loading State
│   │   │   └── SuccessOverlay.tsx          # Success Confirm
│   │   │
│   │   └── 📁 ui/
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── input.tsx
│   │       ├── badge.tsx
│   │       ├── checkbox.tsx
│   │       ├── dialog.tsx
│   │       ├── tabs.tsx
│   │       └── skeleton.tsx
│   │
│   ├── 📁 hooks/
│   │   ├── useResumeUpload.ts    # Upload Logic
│   │   └── useRecommendations.ts # Fetch Logic
│   │
│   ├── 📁 services/
│   │   └── api.ts                # API Client
│   │
│   ├── 📁 lib/
│   │   └── utils.ts              # Utilities
│   │
│   └── 📁 types/
│       ├── api.ts                # API Types
│       └── app.ts                # App Types
│
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
├── postcss.config.mjs
└── README.md
```

---

## 📂 Project Structure

The project follows Next.js App Router conventions with a clean, feature-based architecture:

```
frontend-next/
├── public/
│   └── snu-logo.png              # Sister Nivedita University logo
│
├── src/
│   ├── app/
│   │   ├── globals.css           # Global theme variables, animations, dark mode
│   │   ├── layout.tsx            # Root layout wrapper, Header, Sidebar
│   │   ├── page.tsx              # Welcome/Academic Title Page
│   │   └── dashboard/
│   │       └── page.tsx          # Redirects to home (legacy route)
│   │
│   ├── components/
│   │   ├── features/
│   │   │   ├── DashboardWorkspace.tsx      # Main 4-step workflow container
│   │   │   ├── UploadSection.tsx           # Resume file upload component
│   │   │   ├── SkillsList.tsx              # Extracted skills display
│   │   │   ├── AcademicMarksSection.tsx    # CGPA/Percentage input
│   │   │   ├── RecommendationsGrid.tsx     # Internship results grid
│   │   │   └── SkillSuggestions.tsx        # Skill gap recommendations
│   │   │
│   │   ├── layout/
│   │   │   ├── Header.tsx                  # Top navigation bar
│   │   │   └── Sidebar.tsx                 # (Optional side navigation)
│   │   │
│   │   ├── overlays/
│   │   │   ├── SearchingBooksLoader.tsx    # Loading animation during processing
│   │   │   └── SuccessOverlay.tsx          # Success confirmation overlay
│   │   │
│   │   └── ui/                             # Reusable UI primitives
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── input.tsx
│   │       ├── badge.tsx
│   │       ├── checkbox.tsx
│   │       ├── dialog.tsx
│   │       ├── tabs.tsx
│   │       └── skeleton.tsx
│   │
│   ├── hooks/
│   │   ├── useResumeUpload.ts    # Resume upload state management
│   │   └── useRecommendations.ts # Recommendation fetching logic
│   │
│   ├── services/
│   │   └── api.ts                # Backend API client (fetch wrapper)
│   │
│   ├── lib/
│   │   └── utils.ts              # Utility functions, class name merging
│   │
│   └── types/
│       ├── api.ts                # API request/response types
│       └── app.ts                # Application domain types
│
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
├── postcss.config.mjs
└── README.md
```

---

## 🎨 Design System

### 🌈 Color Palette

| Element            | Color                         | CSS Class              |
| ------------------ | ----------------------------- | ---------------------- |
| **Background**     | `#000000`                     | `bg-black`             |
| **Surfaces**       | `rgba(255,255,255,0.03-0.08)` | `bg-white/[0.03-0.08]` |
| **Primary**        | `#ffd700`                     | `text-[#ffd700]`       |
| **Borders**        | `rgba(255,255,255,0.1-0.2)`   | `border-white/10-20`   |
| **Text Primary**   | `#ffffff`                     | `text-white`           |
| **Text Secondary** | `rgba(255,255,255,0.65)`      | `text-white/65`        |
| **Text Tertiary**  | `rgba(255,255,255,0.45)`      | `text-white/45`        |

### 📐 Spacing & Layout

```
Base Unit:           4px (Tailwind default)
Component Padding:   6-8px internally
Section Gaps:        6-8px between sections
Responsive Grid:     lg:grid-cols-[1.5fr_1fr]
```

### 🔤 Typography

```
Font Family:     Inter (Next.js next/font/google)
Base Size:       16px (1rem)
Heading Scale:   1.25x – 4x
Line Height:     1.5 – 1.75 (readable)
```

---

## 🔗 Backend Integration

### API Configuration

The backend connection is configured in [src/services/api.ts](src/services/api.ts):

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
```

### Available Endpoints

| Method   | Endpoint     | Purpose                               | Status    |
| -------- | ------------ | ------------------------------------- | --------- |
| **GET**  | `/health`    | Backend availability check            | ✅ Active |
| **POST** | `/upload`    | Resume file upload & skill extraction | ✅ Active |
| **POST** | `/recommend` | Generate internship recommendations   | ✅ Active |

### Backend Setup

**Make sure the Python FastAPI backend is running:**

```bash
# From project root
cd backend
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080
```

✨ **Backend must be running for the app to function!**

---

## Responsive Breakpoints

The application is fully responsive and tested on:

- **Mobile:** 320px–768px (Small phones to tablets)
- **Tablet:** 768px–1024px
- **Desktop:** 1024px–1440px
- **Large Screens:** 1440px+ (Ultra-wide displays)

All components use Tailwind's responsive prefixes (`sm:`, `md:`, `lg:`, `xl:`).

---

## ♿ Accessibility Features

- **ARIA Labels:** All interactive elements have descriptive labels
- **Semantic HTML:** Proper heading hierarchy and structure
- **Keyboard Navigation:** Full keyboard support for all controls
- **Color Contrast:** WCAG AA compliant text contrast ratios
- **Focus Management:** Clear visual focus indicators
- **Screen Reader Support:** Content properly structured for assistive technologies

---

## 🎓 Project Information

| Detail         | Information                                                     |
| :------------- | :-------------------------------------------------------------- |
| **University** | Sister Nivedita University (SNU)                                |
| **Program**    | Bachelor of Technology (BTech) - Computer Science & Engineering |
| **Supervisor** | Dr. Sayani Mondal (Assistant Professor)                         |
| **Submission** | November 25, 2025                                               |
| **Session**    | Academic Year 2025                                              |

### Developed By

| Name              | Registration | Email                        |
| :---------------- | :----------- | :--------------------------- |
| Priyangshu Mondal | 220100663543 | mondalpriyangshu@gmail.com   |
| Abhijit Biswas    | 220100017663 | abhijit.biswas1024@gmail.com |
| Kunal Roy         | 220100185465 | royku321@gmail.com           |
| Rupam Haldar      | 220100408950 | prabirhaldar68@gmail.com     |

---

## 📄 License

This project is created as part of an academic Final Year Project and is intended for educational purposes.

---

<p align="center">
  <strong>Built with ❤️ using Next.js 16, Tailwind CSS v4, and TypeScript</strong>
  <br />
  <em>AI Powered Career Guidance System - Frontend Interface</em>
</p>
