import React from "react";
import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { DashboardWorkspace } from "@/components/features/DashboardWorkspace";
import { GraduationCap, LayoutDashboard, Sparkles } from "lucide-react";

export default function Home() {
  return (
    <div className="space-y-8 lg:space-y-10">
      <section className="overflow-hidden rounded-[2rem] border border-slate-200 bg-white p-4 shadow-[0_24px_70px_rgba(15,23,42,0.08)] sm:p-6 md:p-8">
        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
          <div className="space-y-5">
            <div className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-sky-50 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-sky-800">
              <LayoutDashboard className="h-3.5 w-3.5 text-sky-600" />
              AI Powered Career Guidance
            </div>
            <div className="space-y-3">
              <h1 className="max-w-4xl text-3xl font-black uppercase tracking-[0.08em] text-slate-900 sm:text-4xl md:text-5xl lg:text-6xl">
                Resume Analyser and Internship Recommendation System
              </h1>
              <p className="max-w-3xl text-sm leading-7 text-slate-600 sm:text-base md:text-lg">
                This project reads a resume, pulls out the important details,
                and suggests internships that fit the profile. The layout keeps
                the academic summary and the dashboard in one place so the flow
                stays simple.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <Link href="#dashboard">
                <Button className="gap-2">
                  <Sparkles className="h-4 w-4" />
                  Jump to Dashboard
                </Button>
              </Link>
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2 text-sm text-emerald-700">
                <GraduationCap className="h-4 w-4 text-emerald-600" />
                Bachelor of Technology (CSE)
              </div>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
            <div className="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-4">
              <div className="relative mx-auto mb-5 h-28 w-full max-w-sm overflow-hidden rounded-[1rem] border border-slate-200 bg-white">
                <Image
                  src="/snu-logo.png"
                  alt="Sister Nivedita University logo"
                  fill
                  className="object-fill"
                  priority
                />
              </div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                Project Scope
              </p>
              <p className="mt-2 text-sm text-slate-600">
                This system bridges the gap between academic skills and industry
                expectations by providing tailored internship recommendations
                based on resume parsing.
              </p>
            </div>

            <div className="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                Core Methodology
              </p>
              <p className="mt-2 text-sm text-slate-600">
                Utilizing natural language processing to extract core
                competencies, the engine matches candidates with relevant roles
                while highlighting areas for upskilling.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* <section className="space-y-4 rounded-[2rem] border border-white/10 bg-white/[0.03] p-4 shadow-[0_24px_70px_rgba(0,0,0,0.3)] sm:p-6 md:p-8">
        <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-white/70">
          <UserSquare2 className="h-4 w-4 text-[#ffd700]" />
          Submitted By
        </h3>
        <div className="overflow-hidden rounded-[1.25rem] border border-white/10">
          <div className="hidden bg-white/[0.04] px-4 py-3 text-xs font-semibold uppercase tracking-[0.18em] text-white/55 md:grid md:grid-cols-[1.4fr_0.9fr_1.2fr]">
            <div>Student Name</div>
            <div>Reg Number</div>
            <div>Email ID</div>
          </div>
          <div className="divide-y divide-white/10 bg-black/35">
            {[
              {
                name: "Priyangshu Mondal",
                reg: "220100663543",
                email: "mondalpriyangshu@gmail.com",
              },
              {
                name: "Abhijit Biswas",
                reg: "220100017663",
                email: "abhijit.biswas1024@gmail.com",
              },
              {
                name: "Kunal Roy",
                reg: "220100185465",
                email: "royku321@gmail.com",
              },
              {
                name: "Rupam Haldar",
                reg: "220100408950",
                email: "prabirhaldar68@gmail.com",
              },
            ].map((student) => (
              <div
                key={student.reg}
                className="grid gap-2 p-4 hover:bg-white/[0.03] md:grid-cols-[1.4fr_0.9fr_1.2fr] md:items-center"
              >
                <div>
                  <div className="mb-1 text-xs font-semibold uppercase tracking-[0.16em] text-white/40 md:hidden">
                    Student Name
                  </div>
                  <div className="font-medium text-white">{student.name}</div>
                </div>
                <div>
                  <div className="mb-1 text-xs font-semibold uppercase tracking-[0.16em] text-white/40 md:hidden">
                    Reg Number
                  </div>
                  <div className="text-white/70">{student.reg}</div>
                </div>
                <div className="flex items-center gap-2 text-white/70">
                  <Mail className="h-4 w-4 shrink-0 text-[#ffd700]" />
                  <span className="break-all">{student.email}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded-[1.5rem] border border-white/10 bg-white/[0.03] p-5 shadow-[0_20px_60px_rgba(0,0,0,0.28)]">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-bold uppercase tracking-[0.18em] text-white/70">
            <ShieldCheck className="h-4 w-4 text-[#ffd700]" />
            Under the Supervision of
          </h3>
          <p className="text-xl font-semibold text-white">Dr. Sayani Mondal</p>
          <p className="text-sm text-white/60">Assistant Professor</p>
        </div>

        <div className="rounded-[1.5rem] border border-white/10 bg-white/[0.03] p-5 shadow-[0_20px_60px_rgba(0,0,0,0.28)]">
          <h3 className="mb-2 flex items-center gap-2 text-sm font-bold uppercase tracking-[0.18em] text-white/70">
            <CalendarDays className="h-4 w-4 text-[#ffd700]" />
            Submission Date
          </h3>
          <p className="text-xl font-semibold text-white">May, 2026</p>
          <p className="text-sm text-white/60">Academic Session 2022-2026</p>
        </div>
      </section> */}

      <div className="flex justify-center">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">
          Project showcase - May, 2026
        </p>
      </div>

      <DashboardWorkspace />
    </div>
  );
}
