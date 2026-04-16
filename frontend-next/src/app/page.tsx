"use client";

import React from "react";
import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import {
  CalendarDays,
  GraduationCap,
  Mail,
  ShieldCheck,
  UserSquare2,
} from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen px-3 py-6 sm:px-4 sm:py-8 md:px-6 lg:px-8 lg:py-10">
      <div className="mx-auto max-w-5xl app-surface px-4 py-6 sm:px-5 sm:py-8 md:px-8 md:py-10 lg:px-10 lg:py-12">
        <header className="border-b pb-6 text-center sm:pb-8">
          <div className="relative mx-auto mb-6 h-30 w-full max-w-xl overflow-hidden rounded-md bg-white">
            <Image
              src="/snu-logo.png"
              alt="Sister Nivedita University logo"
              fill
              className="object-fill"
              priority
            />
          </div>
          <h1 className="mx-auto max-w-4xl text-xl font-bold uppercase tracking-wide text-[#1a2333] sm:text-2xl md:text-3xl lg:text-4xl">
            AI Powered Career Guidance:
            <br />
            Resume Analyser And Internship Recommendation System
          </h1>
          <div className="mx-auto my-5 h-[2px] w-20 bg-black" />
          <h2 className="text-base font-semibold text-[#111111] sm:text-lg md:text-xl">
            Bachelor of Technology (CSE)
          </h2>
        </header>

        <section className="py-8">
          <h3 className="section-title mb-4 flex items-center justify-center gap-2 text-lg font-bold uppercase">
            <UserSquare2 className="h-5 w-5" />
            Submitted By
          </h3>
          <div className="overflow-x-auto rounded-lg border border-[#d4d4d4]">
            <div className="grid w-full min-w-0 grid-cols-1 border-collapse bg-white text-sm md:table md:min-w-[600px] md:text-base lg:min-w-[680px]">
              <div className="hidden md:table-header-group">
                <div className="table-row">
                  <div className="table-cell border-b border-[#d4d4d4] bg-[#f1f1f1] px-4 py-3 text-left font-bold text-[#111111]">
                    Student Name
                  </div>
                  <div className="table-cell border-b border-[#d4d4d4] bg-[#f1f1f1] px-4 py-3 text-left font-bold text-[#111111]">
                    Reg Number
                  </div>
                  <div className="table-cell border-b border-[#d4d4d4] bg-[#f1f1f1] px-4 py-3 text-left font-bold text-[#111111]">
                    Email ID
                  </div>
                </div>
              </div>
              <div className="table-row-group">
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
                ].map((student, i) => (
                  <div
                    key={i}
                    className="flex flex-col border-b border-[#d4d4d4] p-3 last:border-0 hover:bg-[#f7f7f7] sm:p-4 md:table-row md:p-0"
                  >
                    <div className="mb-1 text-sm font-bold text-[#111111] md:hidden">
                      Student Name
                    </div>
                    <div className="table-cell px-0 py-1 md:border-b md:border-[#d4d4d4] md:px-4 md:py-3 font-medium text-[#1a2333] md:font-normal">
                      {student.name}
                    </div>
                    <div className="mb-1 mt-3 text-sm font-bold text-[#111111] md:hidden">
                      Reg Number
                    </div>
                    <div className="table-cell px-0 py-1 md:border-b md:border-[#d4d4d4] md:px-4 md:py-3 text-[#4a4a4a] md:text-[#1a2333]">
                      {student.reg}
                    </div>
                    <div className="mb-1 mt-3 text-sm font-bold text-[#111111] md:hidden">
                      Email ID
                    </div>
                    <div className="table-cell px-0 py-1 md:border-b md:border-[#d4d4d4] md:px-4 md:py-3">
                      <span className="inline-flex items-center gap-2 break-all text-[#111111] md:text-[#1a2333]">
                        <Mail className="h-4 w-4 shrink-0 text-[#111111]" />
                        {student.email}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <div className="h-px w-full bg-[#d4d4d4]" />

        <section className="grid gap-4 py-6 sm:gap-6 sm:py-8 md:grid-cols-2 md:gap-8">
          <div className="rounded-xl border border-[#d4d4d4] bg-[#f8f8f8] p-5">
            <h3 className="section-title mb-2 flex items-center gap-2 text-sm font-bold uppercase">
              <ShieldCheck className="h-4 w-4" />
              Under the Supervision of
            </h3>
            <p className="text-xl font-semibold text-[#1a2333]">
              Dr. Sayani Mondal
            </p>
            <p className="text-sm text-[#4a4a4a]">Assistant Professor</p>
          </div>

          <div className="rounded-xl border border-[#d4d4d4] bg-[#f8f8f8] p-5">
            <h3 className="section-title mb-2 flex items-center gap-2 text-sm font-bold uppercase">
              <CalendarDays className="h-4 w-4" />
              Submission Date
            </h3>
            <p className="text-xl font-semibold text-[#1a2333]">
              November 25, 2025
            </p>
            <p className="text-sm text-[#4a4a4a]">Academic Session 2025</p>
          </div>
        </section>

        <footer className="border-t border-[#d4d4d4] pt-5 text-center">
          <p className="text-sm font-semibold uppercase tracking-[0.15em] text-[#111111]">
            November 2025
          </p>
        </footer>
      </div>

      <div className="no-print mx-auto mt-5 flex max-w-5xl flex-wrap justify-center gap-3">
        {/* <Button onClick={handlePrint} variant="outline" className="gap-2">
          <Printer className="h-4 w-4" />
          Download as PDF
        </Button> */}
        <Link href="/dashboard">
          <Button className="gap-2">
            <GraduationCap className="h-4 w-4" />
            Open Project Dashboard
          </Button>
        </Link>
      </div>
    </div>
  );
}
