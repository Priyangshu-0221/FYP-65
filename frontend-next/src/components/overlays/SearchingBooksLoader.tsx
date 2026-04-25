"use client";

import { BookOpenText, Search } from "lucide-react";

interface SearchingBooksLoaderProps {
  isVisible: boolean;
  mode?: "upload" | "recommendation";
}

export function SearchingBooksLoader({
  isVisible,
  mode = "upload",
}: SearchingBooksLoaderProps) {
  if (!isVisible) {
    return null;
  }

  const title =
    mode === "upload"
      ? "Searching through books for your resume"
      : "Searching through books for the best internships";

  const description =
    mode === "upload"
      ? "Extracting skills, education, and experience from your PDF."
      : "Matching your profile against internship opportunities.";

  return (
    <div
      aria-live="polite"
      aria-busy="true"
      className="fixed inset-0 z-[200] flex items-center justify-center bg-white/85 px-3 backdrop-blur-md sm:px-4"
    >
      <div className="search-books-panel w-full max-w-[min(92vw,36rem)] rounded-[1.5rem] border border-slate-200 bg-white p-4 text-center text-slate-900 shadow-2xl sm:rounded-3xl sm:p-6">
        <div className="search-books-stage relative mx-auto flex h-[clamp(16rem,55vw,18rem)] w-full items-center justify-center overflow-hidden rounded-[1.5rem] border border-slate-200 bg-[radial-gradient(circle_at_top,rgba(37,99,235,0.12),transparent_45%),linear-gradient(180deg,rgba(16,185,129,0.08),rgba(255,255,255,0.96))] px-3 sm:h-[clamp(18rem,48vw,20rem)] sm:px-4">
          <div className="search-books-shelf absolute bottom-[clamp(1.5rem,4vw,2rem)] left-1/2 h-2 w-[clamp(14rem,78vw,18rem)] -translate-x-1/2 rounded-full bg-slate-200 sm:w-[clamp(16rem,70vw,22rem)]" />

          <div className="search-books-flight search-books-flight-left absolute left-[clamp(0.75rem,4vw,2rem)] top-[clamp(1.5rem,5vw,3.5rem)] h-[clamp(3.25rem,10vw,4.5rem)] w-[clamp(2.1rem,7vw,3rem)] rounded-xl border border-sky-100 bg-white" />
          <div className="search-books-flight search-books-flight-right absolute right-[clamp(0.75rem,4vw,2rem)] top-[clamp(2rem,6vw,4.5rem)] h-[clamp(3.75rem,11vw,5rem)] w-[clamp(2.1rem,7vw,3rem)] rounded-xl border border-emerald-100 bg-white" />

          <div className="search-books-open-book relative z-10 flex h-[clamp(9.5rem,33vw,10.75rem)] w-[clamp(15rem,58vw,18rem)] items-center justify-center">
            <span className="search-books-cover search-books-cover-left absolute left-0 top-[clamp(1rem,3vw,1.4rem)] h-[clamp(5.5rem,18vw,7rem)] w-[clamp(5.5rem,18vw,7rem)] rounded-l-3xl border border-sky-100 bg-gradient-to-br from-sky-500 via-sky-600 to-sky-700 shadow-[0_16px_30px_rgba(37,99,235,0.25)]" />
            <span className="search-books-cover search-books-cover-right absolute right-0 top-[clamp(1rem,3vw,1.4rem)] h-[clamp(5.5rem,18vw,7rem)] w-[clamp(5.5rem,18vw,7rem)] rounded-r-3xl border border-emerald-100 bg-gradient-to-br from-emerald-50 via-white to-emerald-100 shadow-[0_16px_30px_rgba(15,23,42,0.08)]" />
            <span className="search-books-spine absolute left-1/2 top-[clamp(1.65rem,5vw,2.1rem)] h-[clamp(4.6rem,15vw,5.75rem)] w-[clamp(0.55rem,1.5vw,1rem)] -translate-x-1/2 rounded-full bg-slate-900 shadow-[0_0_0_1px_rgba(255,255,255,0.55)]" />
            <span className="search-books-page search-books-page-1 absolute left-[calc(50%-0.4rem)] top-[clamp(1.7rem,5vw,2.15rem)] h-[clamp(4.6rem,15vw,5.75rem)] w-[clamp(2.3rem,7vw,3rem)] origin-left rounded-l-2xl bg-white" />
            <span className="search-books-page search-books-page-2 absolute left-[calc(50%+0.4rem)] top-[clamp(1.7rem,5vw,2.15rem)] h-[clamp(4.6rem,15vw,5.75rem)] w-[clamp(2.3rem,7vw,3rem)] origin-right rounded-r-2xl bg-slate-100" />
            <div className="search-books-search absolute right-[clamp(0.6rem,2vw,1.2rem)] top-[clamp(0.5rem,2vw,1rem)]">
              <Search className="h-[clamp(1.2rem,4vw,1.75rem)] w-[clamp(1.2rem,4vw,1.75rem)] text-sky-600" />
            </div>
            <BookOpenText className="search-books-book absolute bottom-[clamp(0.5rem,2vw,1rem)] h-[clamp(1.9rem,6vw,2.5rem)] w-[clamp(1.9rem,6vw,2.5rem)] text-emerald-600" />
          </div>

          <div className="search-books-scanline absolute inset-x-[clamp(0.75rem,4vw,2rem)] top-1/2 h-px bg-gradient-to-r from-transparent via-sky-500 to-emerald-500 opacity-80" />
          <div className="search-books-scan-dot absolute left-[clamp(0.8rem,3vw,2rem)] top-1/2 h-[clamp(0.5rem,1.4vw,0.8rem)] w-[clamp(0.5rem,1.4vw,0.8rem)] -translate-y-1/2 rounded-full bg-sky-500 shadow-[0_0_20px_rgba(37,99,235,0.7)]" />
        </div>

        <div className="mt-6 space-y-2">
          <p className="text-[clamp(1rem,2.8vw,1.125rem)] font-semibold tracking-wide text-slate-900">
            {title}
          </p>
          <p className="text-[clamp(0.85rem,2.1vw,0.95rem)] text-slate-600">
            {description}
          </p>
        </div>

        <div className="mt-8 flex flex-wrap items-end justify-center gap-2 sm:gap-3">
          {[0, 1, 2, 3].map((index) => (
            <span
              key={index}
              className="search-books-tome"
              style={{ animationDelay: `${index * 0.15}s` }}
            />
          ))}
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-center gap-2 text-[clamp(0.55rem,1.8vw,0.75rem)] uppercase tracking-[0.24em] text-slate-500 sm:tracking-[0.3em]">
          <span className="h-px w-8 bg-slate-300" />
          Scanning pages
          <span className="h-px w-8 bg-slate-300" />
        </div>
      </div>
    </div>
  );
}
