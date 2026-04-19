"use client";

import React, { useEffect, useRef, useState } from "react";
import { UploadSection } from "@/components/features/UploadSection";
import { SkillsList } from "@/components/features/SkillsList";
import { AcademicMarksSection } from "@/components/features/AcademicMarksSection";
import { RecommendationsGrid } from "@/components/features/RecommendationsGrid";
import { SkillSuggestions } from "@/components/features/SkillSuggestions";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useResumeUpload } from "@/hooks/useResumeUpload";
import { useRecommendations } from "@/hooks/useRecommendations";
import { SearchingBooksLoader } from "@/components/overlays/SearchingBooksLoader";
import { SuccessOverlay } from "@/components/overlays/SuccessOverlay";
import type { AcademicMarks } from "@/types/app";
import { ArrowRight, Eraser, Sparkles, Workflow } from "lucide-react";
import { toast } from "react-toastify";

export function DashboardWorkspace() {
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
    recommendedSkills,
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
    if (results.recommendations.length > 0) {
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

  const handleResumeUpload = async (file: File) => {
    if (uploadTimerRef.current) {
      window.clearTimeout(uploadTimerRef.current);
      uploadTimerRef.current = null;
    }

    uploadStartedAtRef.current = Date.now();
    setShowUploadLoader(true);
    try {
      return await handleUpload(file);
    } finally {
      const startedAt = uploadStartedAtRef.current;
      if (startedAt === null) {
        return;
      }

      const elapsed = Date.now() - startedAt;
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
    }
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

  const showBookLoader = showUploadLoader || isRecommending;
  const loaderMode: "upload" | "recommendation" = showUploadLoader
    ? "upload"
    : "recommendation";

  const progressStep =
    skills.length > 0 ? (recommendations.length > 0 ? 4 : 3) : 1;

  return (
    <section
      id="dashboard"
      className="relative space-y-8 overflow-hidden rounded-[2.5rem] border border-white/10 bg-gradient-to-br from-white/[0.05] to-white/[0.01] p-6 shadow-[0_24px_70px_rgba(0,0,0,0.35)] sm:p-8 md:p-10"
    >
      <SearchingBooksLoader isVisible={showBookLoader} mode={loaderMode} />
      <SuccessOverlay isVisible={showRecommendationSuccess} />

      {/* Header Section */}
      <div className="space-y-4 border-b border-white/10 pb-8">
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/[0.08] px-3.5 py-1.5">
              <Workflow className="h-3.5 w-3.5 text-[#ffd700]" />
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-white/70">
                Interactive Workflow
              </p>
            </div>
            <h2 className="text-3xl font-black uppercase tracking-[0.08em] text-white sm:text-5xl">
              Recommendation Engine
            </h2>
            <p className="max-w-2xl text-base leading-7 text-white/60">
              Follow the guided workflow: upload your resume, review extracted
              skills, add academic performance, and discover tailored internship
              opportunities.
            </p>
          </div>
        </div>

        {/* Progress Indicator */}
        <div className="flex items-center gap-2 pt-4">
          {[1, 2, 3, 4].map((step) => (
            <div key={step} className="flex items-center gap-2">
              <div
                className={`h-8 w-8 flex items-center justify-center rounded-full font-semibold text-xs transition-all ${
                  step <= progressStep
                    ? "bg-[#ffd700] text-black"
                    : "border border-white/20 bg-white/[0.05] text-white/50"
                }`}
              >
                {step}
              </div>
              {step < 4 && (
                <div
                  className={`h-0.5 w-8 ${
                    step < progressStep ? "bg-[#ffd700]" : "bg-white/10"
                  }`}
                />
              )}
            </div>
          ))}
          <span className="ml-4 text-xs font-semibold text-white/50 uppercase tracking-wider">
            Step {progressStep} of 4
          </span>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr]">
        {/* Left Column - Steps */}
        <div className="space-y-6">
          {/* Step 1: Upload */}
          <div className="group rounded-2xl border border-white/10 bg-gradient-to-br from-white/[0.06] to-white/[0.02] p-6 transition-all hover:border-white/20 sm:p-8">
            <div className="mb-6 flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-[#ffd700] to-[#ffed4e] text-black">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#ffd700]">
                  Step 1
                </p>
                <h3 className="text-xl font-bold text-white">Upload Resume</h3>
              </div>
            </div>
            <UploadSection
              onUpload={handleResumeUpload}
              isUploading={isUploading}
              fileName={file?.name}
            />
          </div>

          {/* Step 2: Skills */}
          {skills.length > 0 && (
            <div className="group rounded-2xl border border-white/10 bg-gradient-to-br from-white/[0.06] to-white/[0.02] p-6 transition-all hover:border-white/20 sm:p-8">
              <div className="mb-6 flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-[#ffd700] to-[#ffed4e] text-black">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#ffd700]">
                    Step 2
                  </p>
                  <h3 className="text-xl font-bold text-white">
                    Extracted Skills
                  </h3>
                </div>
              </div>
              <SkillsList skills={skills} isLoading={false} />
            </div>
          )}

          {/* Step 3: Marks */}
          {skills.length > 0 && (
            <div className="group rounded-2xl border border-white/10 bg-gradient-to-br from-white/[0.06] to-white/[0.02] p-6 transition-all hover:border-white/20 sm:p-8">
              <div className="mb-6 flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-[#ffd700] to-[#ffed4e] text-black">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#ffd700]">
                    Step 3
                  </p>
                  <h3 className="text-xl font-bold text-white">
                    Academic Profile
                  </h3>
                </div>
              </div>
              <AcademicMarksSection onMarksChange={handleMarksChange} />
            </div>
          )}

          {/* Step 4: Generate Recommendations */}
          {skills.length > 0 && (
            <div className="group rounded-2xl border border-white/10 bg-gradient-to-br from-white/[0.06] to-white/[0.02] p-6 transition-all hover:border-white/20 sm:p-8">
              <div className="mb-6 flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-[#ffd700] to-[#ffed4e] text-black">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#ffd700]">
                    Step 4
                  </p>
                  <h3 className="text-xl font-bold text-white">
                    Generate Matches
                  </h3>
                </div>
              </div>
              <div className="flex flex-col gap-3 sm:flex-row sm:gap-4">
                <Button
                  onClick={handleGetRecommendations}
                  disabled={isRecommending}
                  className="gap-2 px-8 py-3 text-base font-semibold"
                  size="lg"
                >
                  <Sparkles className="h-5 w-5" />
                  {isRecommending
                    ? "Searching Matches..."
                    : "Get Recommendations"}
                  {!isRecommending && <ArrowRight className="h-4 w-4" />}
                </Button>
                {recommendations.length > 0 && (
                  <Button
                    onClick={handleStartOver}
                    variant="outline"
                    size="lg"
                    className="gap-2 border-white/20 px-8 py-3 text-base font-semibold"
                  >
                    <Eraser className="h-4 w-4" />
                    Reset
                  </Button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Status & Results */}
        <div className="space-y-6">
          {/* Status Card */}
          <div className="sticky top-8 rounded-2xl border border-white/10 bg-gradient-to-br from-white/[0.08] to-white/[0.03] p-6 sm:p-8">
            <div className="mb-6 space-y-1">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[#ffd700]">
                Current Status
              </p>
              <h3 className="text-2xl font-bold text-white">
                Processing Overview
              </h3>
            </div>
            <p className="mb-6 text-sm leading-6 text-white/60">
              Real-time tracking of your resume analysis and profile completion.
            </p>

            <div className="space-y-4">
              <div className="rounded-xl border border-white/10 bg-white/[0.05] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/50 mb-2">
                  Resume File
                </p>
                <p className="text-sm font-medium text-white truncate">
                  {file?.name || (
                    <span className="text-white/40">No file uploaded</span>
                  )}
                </p>
              </div>

              <div className="rounded-xl border border-white/10 bg-white/[0.05] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/50 mb-2">
                  Skills Extracted
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-[#ffd700]">
                    {skills.length}
                  </span>
                  <span className="text-xs text-white/50">competencies</span>
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-white/[0.05] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.18em] text-white/50 mb-2">
                  Matches Found
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-[#ffd700]">
                    {recommendations.length}
                  </span>
                  <span className="text-xs text-white/50">opportunities</span>
                </div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          {(recommendations.length > 0 || isRecommending) && (
            <RecommendationsGrid
              recommendations={recommendations}
              isLoading={isRecommending}
            />
          )}

          {/* Skill Suggestions */}
          <SkillSuggestions skills={recommendedSkills} />
        </div>
      </div>
    </section>
  );
}
