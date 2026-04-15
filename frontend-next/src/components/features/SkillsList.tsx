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
      <Card className="app-surface border-[#d8e0ed] p-6">
        <h2 className="mb-4 flex items-center gap-2 text-xl font-bold text-[#1d3b72]">
          <Target className="h-5 w-5" />
          Extracted Skills
        </h2>
        <div className="flex flex-wrap gap-2">
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
    <Card className="app-surface border-[#d8e0ed] p-6">
      <div className="space-y-4">
        <div>
          <h2 className="mb-2 flex items-center gap-2 text-xl font-bold text-[#1d3b72]">
            <Target className="h-5 w-5" />
            Extracted Skills ({skills.length})
          </h2>
          <p className="text-sm text-[#5a687d]">
            Skills detected from your resume
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {skills.map((skill) => (
            <Badge
              key={skill}
              variant="secondary"
              className="cursor-pointer border border-[#c8d4e9] bg-[#eef3fb] px-3 py-1 text-sm font-medium text-[#22314f] transition-all hover:-translate-y-0.5 hover:bg-[#e5ecf8]"
            >
              {skill}
            </Badge>
          ))}
        </div>

        <div className="text-xs text-[#7484a0]">
          Found {skills.length} skill{skills.length !== 1 ? "s" : ""} in your
          resume
        </div>
      </div>
    </Card>
  );
}
