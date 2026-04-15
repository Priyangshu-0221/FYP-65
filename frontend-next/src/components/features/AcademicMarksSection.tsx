"use client";

import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import type { AcademicMarks } from "@/types/app";
import { GraduationCap } from "lucide-react";

interface AcademicMarksSectionProps {
  onMarksChange: (marks: AcademicMarks) => void;
}

export function AcademicMarksSection({
  onMarksChange,
}: AcademicMarksSectionProps) {
  const [cgpa, setCgpa] = useState<string>("");
  const [percentage, setPercentage] = useState<string>("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const marks: AcademicMarks = {};
    if (cgpa) marks.cgpa = parseFloat(cgpa);
    if (percentage) marks.percentage = parseFloat(percentage);
    onMarksChange(marks);
  };

  return (
    <Card className="app-surface border-[#d8e0ed] p-6">
      <div className="space-y-4">
        <div>
          <h2 className="mb-2 flex items-center gap-2 text-xl font-bold text-[#1d3b72]">
            <GraduationCap className="h-5 w-5" />
            Academic Information
          </h2>
          <p className="text-sm text-[#5a687d]">
            (Optional) Add your academic marks for better recommendations
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="mb-2 block text-sm font-medium text-[#33435f]">
                CGPA
              </label>
              <Input
                type="number"
                step="0.01"
                min="0"
                max="10"
                placeholder="e.g., 3.5"
                value={cgpa}
                onChange={(e) => setCgpa(e.target.value)}
                className="bg-white"
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium text-[#33435f]">
                Percentage
              </label>
              <Input
                type="number"
                step="0.1"
                min="0"
                max="100"
                placeholder="e.g., 85.5"
                value={percentage}
                onChange={(e) => setPercentage(e.target.value)}
                className="bg-white"
              />
            </div>
          </div>

          <Button type="submit" className="w-full">
            Update Marks
          </Button>
        </form>
      </div>
    </Card>
  );
}
