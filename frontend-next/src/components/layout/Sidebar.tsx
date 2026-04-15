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
    <aside className="hidden w-80 border-r border-[#d8e0ed] bg-[#f8fafe] p-6 lg:block">
      <div className="space-y-8">
        {/* Features Section */}
        <div>
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-[#1d3b72]">
            <BriefcaseBusiness className="h-5 w-5" />
            Features
          </h2>
          <div className="space-y-3">
            {features.map((feature) => (
              <Card
                key={feature.title}
                className="cursor-pointer border-[#d8e0ed] bg-white p-4 transition-all hover:-translate-y-0.5 hover:shadow-md"
              >
                <div className="flex gap-3">
                  <span className="inline-flex h-9 w-9 items-center justify-center rounded-md bg-[#eef3fb] text-[#1d3b72]">
                    <feature.icon className="h-4 w-4" />
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-[#22314f]">
                      {feature.title}
                    </h3>
                    <p className="text-xs text-[#5a687d]">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Team Section */}
        <div>
          <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold text-[#1d3b72]">
            <Users className="h-5 w-5" />
            Project Team
          </h2>
          <div className="space-y-2">
            {teamMembers.map((member) => (
              <div
                key={member.name}
                className="rounded-lg border border-[#d8e0ed] bg-white p-3"
              >
                <p className="text-sm font-semibold text-[#22314f]">
                  {member.name}
                </p>
                <Badge
                  variant="outline"
                  className="mt-1 gap-1 text-xs text-[#425575]"
                >
                  <IdCard className="h-3 w-3" />
                  {member.role}
                </Badge>
              </div>
            ))}
          </div>
        </div>

        {/* Project Info */}
        <div className="border-t border-[#d8e0ed] pt-4">
          <h3 className="mb-2 text-sm font-semibold text-[#1d3b72]">
            AI Powered Career Guidance
          </h3>
          <p className="mb-1 text-xs font-medium text-[#27549f]">
            Bachelor of Technology (CSE)
          </p>
          <p className="mb-3 mt-2 text-xs text-[#5a687d]">
            Resume Analyser And Internship Recommendation System
          </p>
          <div className="space-y-1 border-t border-[#d8e0ed] pt-3">
            <p className="flex items-center gap-1 text-xs font-semibold text-[#22314f]">
              <ShieldCheck className="h-3.5 w-3.5" />
              Supervisor:
            </p>
            <p className="text-xs text-[#5a687d]">
              Dr. Sayani Mondal (Assistant Professor)
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}
