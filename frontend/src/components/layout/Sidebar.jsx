import React from "react";
import { UNIVERSITY_INFO, TEAM_MEMBERS } from "../../constants/data.js";

const Sidebar = () => {
  return (
    <section className="order-2 lg:order-1 relative flex flex-1 flex-col justify-between gap-4 bg-linear-to-br from-slate-800/20 via-gray-800/10 to-transparent p-4 sm:p-6 lg:p-8">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-32 top-32 h-80 w-80 rounded-full bg-blue-500/20 blur-[100px]" />
        <div className="absolute -right-24 bottom-32 h-72 w-72 rounded-full bg-cyan-400/15 blur-[80px]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,rgba(59,130,246,0.1),transparent_50%)]" />
      </div>

      <div className="relative z-10 flex flex-1 flex-col justify-between gap-4">
        <header className="space-y-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
            <div className="relative overflow-hidden rounded-2xl border-2 border-blue-400/30 bg-white/10 p-2 backdrop-blur-xl">
              <img
                src={UNIVERSITY_INFO.logo}
                alt={UNIVERSITY_INFO.name}
                className="h-10 w-10 sm:h-12 sm:w-12 rounded-xl object-cover"
              />
              <div className="absolute inset-0 rounded-2xl bg-linear-to-tr from-blue-500/20 to-cyan-500/20" />
            </div>
            <div className="space-y-1">
              <p className="text-xs sm:text-sm font-bold uppercase tracking-[0.2em] sm:tracking-[0.3em] text-blue-300">
                {UNIVERSITY_INFO.name}
              </p>
              <h1 className="text-2xl sm:text-3xl font-bold leading-tight text-white">
                AI-Powered Internship Matcher
              </h1>
              <p className="text-sm sm:text-base text-blue-200/90">
                Connecting talent with opportunity through intelligence
              </p>
            </div>
          </div>
        </header>

        <div className="space-y-4">
          <div className="space-y-2">
            <h2 className="text-lg sm:text-xl font-semibold text-white">
              How it works
            </h2>
            <p className="text-sm sm:text-base leading-relaxed text-slate-200/80">
              Our advanced AI analyzes your resume to extract key skills and
              matches you with the most relevant internship opportunities.
            </p>
          </div>

          <div className="grid gap-3 sm:gap-4">
            <div className="group rounded-xl border border-blue-400/20 bg-white/5 p-3 sm:p-4 backdrop-blur-sm transition-all hover:bg-white/10">
              <div className="flex items-start gap-3">
                <div className="rounded-full bg-blue-500/20 p-2">
                  <svg
                    className="h-4 w-4 sm:h-5 sm:w-5 text-blue-300"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                </div>
                <div>
                  <p className="text-sm sm:text-base font-semibold text-white">
                    Smart Skill Extraction
                  </p>
                  <p className="text-xs sm:text-sm text-slate-200/70">
                    AI-powered resume parsing identifies your key competencies
                    automatically
                  </p>
                </div>
              </div>
            </div>

            <div className="group rounded-xl border border-cyan-400/20 bg-white/5 p-3 sm:p-4 backdrop-blur-sm transition-all hover:bg-white/10">
              <div className="flex items-start gap-3">
                <div className="rounded-full bg-cyan-500/20 p-2">
                  <svg
                    className="h-4 w-4 sm:h-5 sm:w-5 text-cyan-300"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <div>
                  <p className="text-sm sm:text-base font-semibold text-white">
                    Precision Matching
                  </p>
                  <p className="text-xs sm:text-sm text-slate-200/70">
                    Advanced algorithms find opportunities that align with your
                    unique profile
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-400/20 bg-linear-to-r from-slate-800/30 via-gray-800/20 to-slate-800/30 p-4 backdrop-blur-xl">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between text-sm gap-2">
            <span className="font-bold uppercase tracking-[0.2em] sm:tracking-[0.3em] text-blue-300">
              Development Team
            </span>
            <span className="text-xs sm:text-sm text-slate-200/60">
              {UNIVERSITY_INFO.course}
            </span>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {TEAM_MEMBERS.map((member) => (
              <div
                key={member.id}
                className="group flex items-center gap-2 rounded-xl border border-slate-400/20 bg-white/5 p-2 transition-all hover:bg-white/10"
              >
                <div className="relative">
                  <div className="grid h-6 w-6 sm:h-8 sm:w-8 place-items-center rounded-xl bg-linear-to-br from-blue-500 to-cyan-600 text-xs sm:text-sm font-bold text-white shadow-lg">
                    {member.initial}
                  </div>
                </div>
                <div className="flex-1">
                  <p className="text-sm sm:text-base font-semibold text-white">
                    {member.name}
                  </p>
                  <p className="text-xs sm:text-sm text-slate-200/70">
                    {member.id}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Sidebar;
