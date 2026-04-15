"use client";

import React, { useState } from "react";
import { UploadSection } from "@/components/features/UploadSection";
import { SkillsList } from "@/components/features/SkillsList";
import { AcademicMarksSection } from "@/components/features/AcademicMarksSection";
import { RecommendationsGrid } from "@/components/features/RecommendationsGrid";
import { Button } from "@/components/ui/button";
import { useResumeUpload } from "@/hooks/useResumeUpload";
import { useRecommendations } from "@/hooks/useRecommendations";
import type { AcademicMarks } from "@/types/app";
import { toast } from "sonner";
import { ArrowRight, Eraser, Sparkles, Workflow } from "lucide-react";

function DashboardContent() {
  const [academicMarks, setAcademicMarks] = useState<AcademicMarks>({});

  const {
    file,
    skills,
    isUploading,
    error: uploadError,
    handleUpload,
    resetUpload,
  } = useResumeUpload();

  const {
    recommendations,
    isLoading: isRecommending,
    error: recommendError,
    requestRecommendations,
  } = useRecommendations();

  const handleGetRecommendations = async () => {
    if (!skills || skills.length === 0) {
      toast.error("Please upload a resume first");
      return;
    }
    await requestRecommendations(skills, academicMarks);
  };

  const handleMarksChange = (marks: AcademicMarks) => {
    setAcademicMarks(marks);
    toast.success("Academic marks updated");
  };

  return (
    <div className="min-h-screen px-4 py-8 md:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Welcome Section */}
        <div className="app-surface mb-8 border-[#d8e0ed] p-6 md:p-8">
          <h1 className="mb-2 text-3xl font-bold text-[#1d3b72] md:text-4xl">
            Project Dashboard
          </h1>
          <p className="max-w-3xl text-sm text-[#5a687d] md:text-base">
            Upload your resume, verify extracted skills, add academic details,
            and generate AI-powered internship recommendations aligned with your
            profile.
          </p>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-[#c8d4e9] bg-[#eef3fb] px-3 py-1 text-xs font-semibold text-[#1d3b72]">
            <Workflow className="h-3.5 w-3.5" />
            Structured 4-Step Workflow
          </div>
        </div>

        {/* Error Messages */}
        {uploadError && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            Upload Error: {uploadError}
          </div>
        )}
        {recommendError && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            Recommendation Error: {recommendError}
          </div>
        )}

        {/* Steps Layout */}
        <div className="space-y-6">
          {/* Step 1: Upload Resume */}
          <div>
            <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#1d3b72]">
              Step 1 of 4
            </div>
            <UploadSection
              onUpload={handleUpload}
              isUploading={isUploading}
              fileName={file?.name}
            />
          </div>

          {/* Step 2: View Skills */}
          {skills.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#1d3b72]">
                Step 2 of 4
              </div>
              <SkillsList skills={skills} isLoading={false} />
            </div>
          )}

          {/* Step 3: Add Academic Info */}
          {skills.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#1d3b72]">
                Step 3 of 4
              </div>
              <AcademicMarksSection onMarksChange={handleMarksChange} />
            </div>
          )}

          {/* Step 4: Get Recommendations */}
          {skills.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#1d3b72]">
                Step 4 of 4
              </div>
              <div className="flex gap-4">
                <Button
                  onClick={handleGetRecommendations}
                  disabled={isRecommending}
                  className="gap-2 px-8"
                  size="lg"
                >
                  <Sparkles className="h-4 w-4" />
                  {isRecommending
                    ? "Finding Matches..."
                    : "Get Recommendations"}
                  {!isRecommending && <ArrowRight className="h-4 w-4" />}
                </Button>
                {recommendations.length > 0 && (
                  <Button
                    onClick={resetUpload}
                    variant="outline"
                    size="lg"
                    className="gap-2"
                  >
                    <Eraser className="h-4 w-4" />
                    Start Over
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {(recommendations.length > 0 || isRecommending) && (
            <div>
              <RecommendationsGrid
                recommendations={recommendations}
                isLoading={isRecommending}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return <DashboardContent />;
}
