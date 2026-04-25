"use client";

import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import type { AcademicMarks } from "@/types/app";
import { GraduationCap } from "lucide-react";
import { toast } from "react-toastify";

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

    const parsedCgpa = cgpa ? parseFloat(cgpa) : undefined;
    const parsedPercentage = percentage ? parseFloat(percentage) : undefined;

    if (
      parsedCgpa !== undefined &&
      (!Number.isFinite(parsedCgpa) || parsedCgpa < 0 || parsedCgpa > 10)
    ) {
      toast.error("CGPA must be between 0 and 10");
      return;
    }

    if (
      parsedPercentage !== undefined &&
      (!Number.isFinite(parsedPercentage) ||
        parsedPercentage < 0 ||
        parsedPercentage > 100)
    ) {
      toast.error("Percentage must be between 0 and 100");
      return;
    }

    const marks: AcademicMarks = {};
    if (parsedCgpa !== undefined) marks.cgpa = parsedCgpa;
    if (parsedPercentage !== undefined) marks.percentage = parsedPercentage;
    onMarksChange(marks);
  };

  return (
    <Card className="border-slate-200 p-4 sm:p-5 md:p-6">
      <div className="space-y-3 sm:space-y-4">
        <div>
          <h2 className="mb-2 flex items-center gap-2 text-xl font-bold text-slate-900">
            <GraduationCap className="h-5 w-5 text-sky-600" />
            Academic Information
          </h2>
          <p className="text-sm text-slate-600">
            (Optional) Add your academic marks for better recommendations
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 gap-3 sm:gap-4 md:grid-cols-2">
            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
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
                className="bg-slate-50"
              />
            </div>
            <div>
              <label className="mb-2 block text-sm font-medium text-slate-700">
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
                className="bg-slate-50"
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
