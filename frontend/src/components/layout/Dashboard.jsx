import React from "react";
import UploadSection from "../features/UploadSection.jsx";
import SkillsSection from "../features/SkillsSection.jsx";
import RecommendationsSection from "../features/RecommendationsSection.jsx";

const Dashboard = ({
  file,
  skills,
  recommendations,
  isUploading,
  isRecommending,
  onFileChange,
  onUpload,
  onRequestRecommendations,
}) => {
  return (
    <main className="order-1 lg:order-2 flex flex-1 flex-col bg-linear-to-br from-slate-50 via-white to-slate-100 p-4 sm:p-6 text-slate-900">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
        <div className="space-y-1">
          <p className="text-xs sm:text-sm font-bold uppercase tracking-[0.2em] sm:tracking-[0.3em] text-blue-600">
            Interactive Dashboard
          </p>
          <h2 className="text-2xl sm:text-3xl font-bold text-slate-900">
            Upload. Analyze. Connect.
          </h2>
          <p className="text-sm sm:text-base text-slate-600">
            Transform your resume into career opportunities
          </p>
        </div>
        <div className="hidden sm:block text-center lg:block">
          <div className="rounded-xl border border-blue-200 bg-blue-50 p-3">
            <p className="text-xl sm:text-2xl font-bold text-blue-600">6+</p>
            <p className="text-xs sm:text-sm font-medium text-blue-500">
              Smart matches
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-4">
        <UploadSection
          file={file}
          isUploading={isUploading}
          onFileChange={onFileChange}
          onUpload={onUpload}
        />

        <SkillsSection skills={skills} />

        <RecommendationsSection
          recommendations={recommendations}
          isRecommending={isRecommending}
          onRequestRecommendations={onRequestRecommendations}
          skillsLength={skills.length}
        />
      </div>
    </main>
  );
};

export default Dashboard;
