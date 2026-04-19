"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Sparkles, TrendingUp, Info } from "lucide-react";

interface SkillSuggestionsProps {
  skills: string[];
  isLoading?: boolean;
}

export function SkillSuggestions({ skills, isLoading = false }: SkillSuggestionsProps) {
  if (isLoading) return null; // Usually shown after recommendations are loaded

  if (!skills || skills.length === 0) {
    return null;
  }

  return (
    <Card className="app-surface border-[#d4d4d4] p-4 sm:p-5 md:p-6 bg-gradient-to-br from-white to-[#f9f9f9]">
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-xl font-bold text-[#111111]">
              <Sparkles className="h-5 w-5 text-amber-500 fill-amber-500" />
              Skill Gap Analysis
            </h2>
            <p className="text-sm text-[#4a4a4a] mt-1">
              Top skills missing from your CV that are highly valued in recommended roles
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-black px-3 py-1.5 text-xs font-medium text-white shadow-sm">
            <TrendingUp className="h-3.5 w-3.5" />
            High Demand
          </div>
        </div>

        <div className="flex flex-wrap gap-2 sm:gap-3">
          {skills.map((skill) => (
            <div key={skill} className="group relative">
              <Badge
                variant="outline"
                className="cursor-default border-[#d4d4d4] bg-white px-4 py-2 text-sm font-semibold text-[#111111] shadow-sm transition-all hover:border-black hover:bg-[#f1f1f1]"
              >
                {skill}
              </Badge>
            </div>
          ))}
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-blue-100 bg-blue-50/50 p-4 text-xs text-blue-700">
          <Info className="h-4 w-4 mt-0.5 shrink-0" />
          <p className="leading-relaxed">
            <strong>Pro Tip:</strong> Learning these skills and adding them to your resume could increase your match rate by up to <strong>45%</strong> for the internships listed below.
          </p>
        </div>
      </div>
    </Card>
  );
}
