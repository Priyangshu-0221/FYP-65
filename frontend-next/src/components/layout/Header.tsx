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
    <header className="sticky top-0 z-50 border-b border-[#d8e0ed] bg-white/95 backdrop-blur">
      <div className="mx-auto flex h-20 w-full max-w-[1400px] items-center justify-between px-4 md:px-8">
        <div className="flex items-center gap-4 md:gap-10">
          <Link href="/" className="flex items-center gap-2 md:gap-3">
            <span className="inline-flex h-9 w-9 md:h-11 md:w-11 items-center justify-center rounded-full border border-[#c7d4ea] bg-[#eef3fb] text-[#1d3b72]">
              <University className="h-4 w-4 md:h-5 md:w-5" />
            </span>
            <span className="flex flex-col">
              <h1 className="text-sm font-bold tracking-wide text-[#1d3b72] md:text-lg lg:text-xl">
                Career Guidance System
              </h1>
              <p className="hidden text-xs text-[#5a687d] sm:block md:text-sm">
                Resume Analyser & Internship Recommendation
              </p>
            </span>
          </Link>

          <nav className="hidden items-center gap-6 md:flex">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-sm font-medium text-[#33435f] transition-colors hover:text-[#1d3b72]"
            >
              <BookOpenText className="h-4 w-4" />
              Welcome
            </Link>
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 text-sm font-medium text-[#33435f] transition-colors hover:text-[#1d3b72]"
            >
              <LayoutDashboard className="h-4 w-4" />
              Dashboard
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-2 md:gap-3">
          <Link href="/dashboard">
            <Button size="sm" className="gap-2 px-3 md:px-4">
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
