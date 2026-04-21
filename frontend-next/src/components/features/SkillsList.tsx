"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Target } from "lucide-react";

interface SkillsListProps {
  skills: string[];
  isLoading?: boolean;
}

export function SkillsList({ skills, isLoading = false }: SkillsListProps) {
  if (isLoading) {
    return (
      <Card className="border-slate-200 p-4 sm:p-5 md:p-6">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-bold text-slate-900">
          <Target className="h-5 w-5 text-sky-600" />
          Extracted Skills
        </h2>
        <div className="flex flex-wrap gap-1.5 sm:gap-2">
          {[...Array(8)].map((_, i) => (
            <Skeleton key={i} className="h-8 w-20 rounded-full" />
          ))}
        </div>
      </Card>
    );
  }

  if (!skills || skills.length === 0) {
    return null;
  }

  return (
    <Card className="border-slate-200 p-4 sm:p-5 md:p-6">
      <div className="space-y-3 sm:space-y-4">
        <div>
          <h2 className="mb-2 flex items-center gap-2 text-xl font-bold text-slate-900">
            <Target className="h-5 w-5 text-sky-600" />
            Extracted Skills ({skills.length})
          </h2>
          <p className="text-sm text-slate-600">
            Skills detected from your resume
          </p>
        </div>

        <div className="flex flex-wrap gap-1.5 sm:gap-2">
          {skills.map((skill) => (
            <Badge
              key={skill}
              variant="secondary"
              className="cursor-pointer border border-emerald-200 bg-emerald-50 px-3 py-1 text-sm font-medium text-emerald-700 transition-all hover:-translate-y-0.5 hover:bg-emerald-100"
            >
              {skill}
            </Badge>
          ))}
        </div>

        <div className="text-xs text-slate-500">
          Found {skills.length} skill{skills.length !== 1 ? "s" : ""} in your
          resume
        </div>
      </div>
    </Card>
  );
}
