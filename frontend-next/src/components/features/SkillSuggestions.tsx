"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Sparkles, TrendingUp, Info } from "lucide-react";

interface SkillSuggestionsProps {
  skills: string[];
  isLoading?: boolean;
}

export function SkillSuggestions({
  skills,
  isLoading = false,
}: SkillSuggestionsProps) {
  if (isLoading) return null; // Usually shown after recommendations are loaded

  if (!skills || skills.length === 0) {
    return null;
  }

  return (
    <Card className="border-white/10 p-4 sm:p-5 md:p-6">
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-xl font-bold text-white">
              <Sparkles className="h-5 w-5 fill-[#ffd700] text-[#ffd700]" />
              Skill Gap Analysis
            </h2>
            <p className="mt-1 text-sm text-white/65">
              These are the skills that show up in the recommendations but are
              still missing from the resume.
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-[#ffd700] px-3 py-1.5 text-xs font-medium text-black shadow-sm">
            <TrendingUp className="h-3.5 w-3.5" />
            High Demand
          </div>
        </div>

        <div className="flex flex-wrap gap-2 sm:gap-3">
          {skills.map((skill) => (
            <div key={skill} className="group relative">
              <Badge
                variant="outline"
                className="cursor-default border-white/10 bg-white/[0.05] px-4 py-2 text-sm font-semibold text-white shadow-sm transition-all hover:border-white/20 hover:bg-white/[0.08]"
              >
                {skill}
              </Badge>
            </div>
          ))}
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-white/10 bg-white/[0.05] p-4 text-xs text-white/70">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-[#ffd700]" />
          <p className="leading-relaxed">
            <strong className="text-white">Tip:</strong> If you add a few of
            these skills to your resume and can talk about them clearly, you
            will usually get better internship matches.
          </p>
        </div>
      </div>
    </Card>
  );
}
