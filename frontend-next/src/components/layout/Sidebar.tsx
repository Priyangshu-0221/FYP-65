"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BriefcaseBusiness,
  ClipboardCheck,
  Cpu,
  GraduationCap,
  IdCard,
  ShieldCheck,
  Sparkles,
  Users,
} from "lucide-react";

export function Sidebar() {
  const features = [
    {
      title: "Upload Resume",
      description: "Extract skills and information from your resume",
      icon: ClipboardCheck,
    },
    {
      title: "Smart Matching",
      description: "AI-powered matching based on your skills",
      icon: Cpu,
    },
    {
      title: "Recommendations",
      description: "Get personalized internship recommendations",
      icon: Sparkles,
    },
    {
      title: "Academic Info",
      description: "Include your academic marks for better matches",
      icon: GraduationCap,
    },
  ];

  const teamMembers = [
    { name: "Priyangshu Mondal", role: "Reg: 220100663543" },
    { name: "Abhijit Biswas", role: "Reg: 220100017663" },
    { name: "Kunal Roy", role: "Reg: 220100185465" },
    { name: "Rupam Haldar", role: "Reg: 220100408950" },
  ];

  return (
    <aside className="border-b border-slate-200 bg-white">
      <div className="mx-auto grid w-full max-w-[1400px] gap-4 px-3 py-4 sm:px-4 md:px-8 xl:grid-cols-[1.15fr_0.9fr_0.95fr] xl:gap-5 xl:py-5">
        <Card className="border-slate-200 p-4 shadow-[0_20px_50px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-900">
            <BriefcaseBusiness className="h-5 w-5 text-sky-600" />
            Features
          </h2>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
            {features.map((feature) => (
              <Card
                key={feature.title}
                className="cursor-pointer border-slate-200 bg-slate-50 p-4 transition-all hover:-translate-y-0.5 hover:border-sky-200 hover:bg-sky-50"
              >
                <div className="flex gap-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-sky-100 bg-white text-sky-600">
                    <feature.icon className="h-4 w-4" />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900">
                      {feature.title}
                    </h3>
                    <p className="text-xs text-slate-600">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </Card>

        <Card className="border-slate-200 p-4 shadow-[0_20px_50px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-900">
            <Users className="h-5 w-5 text-emerald-600" />
            Project Team
          </h2>
          <div className="space-y-2">
            {teamMembers.map((member) => (
              <div
                key={member.name}
                className="rounded-lg border border-slate-200 bg-slate-50 p-3"
              >
                <p className="text-sm font-semibold text-slate-900">
                  {member.name}
                </p>
                <Badge
                  variant="outline"
                  className="mt-1 gap-1 border-emerald-200 bg-emerald-50 text-xs text-emerald-700"
                >
                  <IdCard className="h-3 w-3" />
                  {member.role}
                </Badge>
              </div>
            ))}
          </div>
        </Card>

        <Card className="border-slate-200 p-4 shadow-[0_20px_50px_rgba(15,23,42,0.08)] backdrop-blur-sm">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.14em] text-slate-900">
            AI Powered Career Guidance
          </h3>
          <p className="mb-1 text-xs font-medium text-slate-600">
            Bachelor of Technology (CSE)
          </p>
          <p className="mb-3 mt-2 text-xs text-slate-600">
            Resume Analyser And Internship Recommendation System
          </p>
          <div className="space-y-1 border-t border-slate-200 pt-3">
            <p className="flex items-center gap-1 text-xs font-semibold text-slate-900">
              <ShieldCheck className="h-3.5 w-3.5 text-sky-600" />
              Supervisor:
            </p>
            <p className="text-xs text-slate-600">
              Dr. Sayani Mondal (Assistant Professor)
            </p>
          </div>
        </Card>
      </div>
    </aside>
  );
}
