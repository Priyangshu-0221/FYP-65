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
    <Card className="border-slate-200 p-4 sm:p-5 md:p-6">
      <div className="space-y-4">
        <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <div>
            <h2 className="flex items-center gap-2 text-xl font-bold text-slate-900">
              <Sparkles className="h-5 w-5 fill-sky-600 text-sky-600" />
              Skill Gap Analysis
            </h2>
            <p className="mt-1 text-sm text-slate-600">
              These are the skills that show up in the recommendations but are
              still missing from the resume.
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 shadow-sm">
            <TrendingUp className="h-3.5 w-3.5" />
            High Demand
          </div>
        </div>

        <div className="flex flex-wrap gap-2 sm:gap-3">
          {skills.map((skill) => (
            <div key={skill} className="group relative">
              <Badge
                variant="outline"
                className="cursor-default border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition-all hover:border-sky-200 hover:bg-sky-50"
              >
                {skill}
              </Badge>
            </div>
          ))}
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-600">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-sky-600" />
          <p className="leading-relaxed">
            <strong className="text-slate-900">Tip:</strong> If you add a few of
            these skills to your resume and can talk about them clearly, you
            will usually get better internship matches.
          </p>
        </div>
      </div>
    </Card>
  );
}
