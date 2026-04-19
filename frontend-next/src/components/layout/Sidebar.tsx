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
    <aside className="border-b border-white/10 bg-white/[0.03]">
      <div className="mx-auto grid w-full max-w-[1400px] gap-4 px-3 py-4 sm:px-4 md:px-8 xl:grid-cols-[1.15fr_0.9fr_0.95fr] xl:gap-5 xl:py-5">
        <Card className="border-white/10 bg-black/55 p-4 shadow-[0_20px_50px_rgba(0,0,0,0.35)] backdrop-blur-sm">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <BriefcaseBusiness className="h-5 w-5 text-[#ffd700]" />
            Features
          </h2>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
            {features.map((feature) => (
              <Card
                key={feature.title}
                className="cursor-pointer border-white/10 bg-white/[0.04] p-4 transition-all hover:-translate-y-0.5 hover:border-white/20 hover:bg-white/[0.07]"
              >
                <div className="flex gap-3">
                  <span className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-white/10 bg-black text-[#ffd700]">
                    <feature.icon className="h-4 w-4" />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-white">
                      {feature.title}
                    </h3>
                    <p className="text-xs text-white/65">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </Card>

        <Card className="border-white/10 bg-black/55 p-4 shadow-[0_20px_50px_rgba(0,0,0,0.35)] backdrop-blur-sm">
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-white">
            <Users className="h-5 w-5 text-[#ffd700]" />
            Project Team
          </h2>
          <div className="space-y-2">
            {teamMembers.map((member) => (
              <div
                key={member.name}
                className="rounded-lg border border-white/10 bg-white/[0.04] p-3"
              >
                <p className="text-sm font-semibold text-white">
                  {member.name}
                </p>
                <Badge
                  variant="outline"
                  className="mt-1 gap-1 border-white/15 bg-white/[0.06] text-xs text-white/80"
                >
                  <IdCard className="h-3 w-3" />
                  {member.role}
                </Badge>
              </div>
            ))}
          </div>
        </Card>

        <Card className="border-white/10 bg-black/55 p-4 shadow-[0_20px_50px_rgba(0,0,0,0.35)] backdrop-blur-sm">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-[0.14em] text-white/90">
            AI Powered Career Guidance
          </h3>
          <p className="mb-1 text-xs font-medium text-white/75">
            Bachelor of Technology (CSE)
          </p>
          <p className="mb-3 mt-2 text-xs text-white/60">
            Resume Analyser And Internship Recommendation System
          </p>
          <div className="space-y-1 border-t border-white/10 pt-3">
            <p className="flex items-center gap-1 text-xs font-semibold text-white">
              <ShieldCheck className="h-3.5 w-3.5 text-[#ffd700]" />
              Supervisor:
            </p>
            <p className="text-xs text-white/65">
              Dr. Sayani Mondal (Assistant Professor)
            </p>
          </div>
        </Card>
      </div>
    </aside>
  );
}
