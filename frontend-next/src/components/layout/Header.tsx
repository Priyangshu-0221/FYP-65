"use client";

import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  BookOpenText,
  GraduationCap,
  LayoutDashboard,
  University,
} from "lucide-react";

export function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-slate-200 bg-white/90 backdrop-blur-xl">
      <div className="mx-auto flex h-16 w-full max-w-[1400px] items-center justify-between px-3 sm:h-20 sm:px-4 md:px-8">
        <div className="flex min-w-0 items-center gap-3 sm:gap-4 md:gap-10">
          <Link href="/" className="flex items-center gap-2 md:gap-3">
            <span className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-sky-200 bg-gradient-to-br from-sky-600 to-emerald-500 text-white md:h-11 md:w-11">
              <University className="h-4 w-4 md:h-5 md:w-5" />
            </span>
            <span className="flex min-w-0 flex-col">
              <h1 className="truncate text-xs font-bold tracking-wide text-slate-900 sm:text-sm md:text-lg lg:text-xl">
                Career Guidance System
              </h1>
              <p className="hidden text-xs text-slate-600 sm:block md:text-sm">
                Resume Analyser & Internship Recommendation
              </p>
            </span>
          </Link>

          <nav className="hidden items-center gap-6 md:flex">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-sm font-medium text-slate-600 transition-colors hover:text-sky-700"
            >
              <BookOpenText className="h-4 w-4" />
              Welcome
            </Link>
            <Link
              href="/#dashboard"
              className="inline-flex items-center gap-2 text-sm font-medium text-slate-600 transition-colors hover:text-sky-700"
            >
              <LayoutDashboard className="h-4 w-4" />
              Dashboard
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-2 md:gap-3">
          <Link href="/#dashboard">
            <Button
              size="sm"
              className="gap-1.5 px-2.5 sm:gap-2 sm:px-3 md:px-4"
            >
              <GraduationCap className="h-4 w-4" />
              <span className="hidden sm:inline">Open Dashboard</span>
              <span className="sm:hidden">Dashboard</span>
            </Button>
          </Link>
        </div>
      </div>
    </header>
  );
}
