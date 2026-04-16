"use client";

import React, { useEffect, useRef, useState } from "react";
import { UploadSection } from "@/components/features/UploadSection";
import { SkillsList } from "@/components/features/SkillsList";
import { AcademicMarksSection } from "@/components/features/AcademicMarksSection";
import { RecommendationsGrid } from "@/components/features/RecommendationsGrid";
import { Button } from "@/components/ui/button";
import { useResumeUpload } from "@/hooks/useResumeUpload";
import { useRecommendations } from "@/hooks/useRecommendations";
import { SearchingBooksLoader } from "@/components/overlays/SearchingBooksLoader";
import { SuccessOverlay } from "@/components/overlays/SuccessOverlay";
import type { AcademicMarks } from "@/types/app";
import { toast } from "react-toastify";
import { ArrowRight, Eraser, Sparkles, Workflow } from "lucide-react";

function DashboardContent() {
  const [academicMarks, setAcademicMarks] = useState<AcademicMarks>({});
  const [showUploadLoader, setShowUploadLoader] = useState(false);
  const [showRecommendationSuccess, setShowRecommendationSuccess] =
    useState(false);
  const uploadTimerRef = useRef<number | null>(null);
  const uploadStartedAtRef = useRef<number | null>(null);
  const successTimerRef = useRef<number | null>(null);

  const { file, skills, isUploading, handleUpload, resetUpload } =
    useResumeUpload();

  const {
    recommendations,
    isLoading: isRecommending,
    requestRecommendations,
    resetRecommendations,
  } = useRecommendations();

  const handleGetRecommendations = async () => {
    if (!skills || skills.length === 0) {
      toast.error("Please upload a resume first");
      return;
    }

    const results = await requestRecommendations(skills, academicMarks);
    if (results.length > 0) {
      if (successTimerRef.current) {
        window.clearTimeout(successTimerRef.current);
      }

      setShowRecommendationSuccess(true);
      successTimerRef.current = window.setTimeout(() => {
        setShowRecommendationSuccess(false);
        successTimerRef.current = null;
      }, 2000);
    }
  };

  const handleMarksChange = (marks: AcademicMarks) => {
    setAcademicMarks(marks);
    toast.success("Academic marks updated");
  };

  const handleStartOver = () => {
    resetUpload();
    resetRecommendations();
    setAcademicMarks({});
    setShowUploadLoader(false);
    setShowRecommendationSuccess(false);
    if (uploadTimerRef.current) {
      window.clearTimeout(uploadTimerRef.current);
      uploadTimerRef.current = null;
    }
    uploadStartedAtRef.current = null;
    if (successTimerRef.current) {
      window.clearTimeout(successTimerRef.current);
      successTimerRef.current = null;
    }
    toast.info("Dashboard reset");
  };

  useEffect(
    () => () => {
      if (uploadTimerRef.current) {
        window.clearTimeout(uploadTimerRef.current);
      }
      if (successTimerRef.current) {
        window.clearTimeout(successTimerRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    if (isUploading) {
      if (uploadTimerRef.current) {
        window.clearTimeout(uploadTimerRef.current);
        uploadTimerRef.current = null;
      }

      if (!showUploadLoader) {
        uploadStartedAtRef.current = Date.now();
        setShowUploadLoader(true);
      }

      return;
    }

    if (!showUploadLoader || uploadStartedAtRef.current === null) {
      return;
    }

    const elapsed = Date.now() - uploadStartedAtRef.current;
    const remaining = Math.max(0, 2000 - elapsed);

    if (remaining === 0) {
      setShowUploadLoader(false);
      uploadStartedAtRef.current = null;
      return;
    }

    uploadTimerRef.current = window.setTimeout(() => {
      setShowUploadLoader(false);
      uploadStartedAtRef.current = null;
      uploadTimerRef.current = null;
    }, remaining);
  }, [isUploading, showUploadLoader]);

  const showBookLoader = showUploadLoader || isRecommending;
  const loaderMode: "upload" | "recommendation" = showUploadLoader
    ? "upload"
    : "recommendation";

  return (
    <div className="relative min-h-screen px-3 py-6 sm:px-4 sm:py-8 md:px-6 lg:px-8 lg:py-10">
      <SearchingBooksLoader isVisible={showBookLoader} mode={loaderMode} />
      <SuccessOverlay isVisible={showRecommendationSuccess} />
      <div className="mx-auto w-full max-w-6xl">
        {/* Welcome Section */}
        <div className="app-surface mb-6 border-[#d4d4d4] p-4 sm:p-6 md:p-8">
          <h1 className="mb-2 text-3xl font-bold text-[#111111] md:text-4xl">
            Project Dashboard
          </h1>
          <p className="max-w-3xl text-sm text-[#4a4a4a] md:text-base">
            Upload your resume, verify extracted skills, add academic details,
            and generate AI-powered internship recommendations aligned with your
            profile.
          </p>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-[#d4d4d4] bg-[#f1f1f1] px-3 py-1 text-xs font-semibold text-[#111111]">
            <Workflow className="h-3.5 w-3.5" />
            Structured 4-Step Workflow
          </div>
        </div>

        {/* Steps Layout */}
        <div className="space-y-4 sm:space-y-6">
          {/* Step 1: Upload Resume */}
          <div>
            <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#111111]">
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
              <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#111111]">
                Step 2 of 4
              </div>
              <SkillsList skills={skills} isLoading={false} />
            </div>
          )}

          {/* Step 3: Add Academic Info */}
          {skills.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#111111]">
                Step 3 of 4
              </div>
              <AcademicMarksSection onMarksChange={handleMarksChange} />
            </div>
          )}

          {/* Step 4: Get Recommendations */}
          {skills.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-[#111111]">
                Step 4 of 4
              </div>
              <div className="flex flex-col gap-3 sm:flex-row sm:gap-4">
                <Button
                  onClick={handleGetRecommendations}
                  disabled={isRecommending}
                  className="w-full gap-2 sm:w-auto sm:px-8"
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
                    onClick={handleStartOver}
                    variant="outline"
                    size="lg"
                    className="w-full gap-2 sm:w-auto"
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
