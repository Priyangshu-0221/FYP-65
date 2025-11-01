import React from "react";
import RecommendationGrid from "../ui/RecommendationGrid.jsx";

const RecommendationsSection = ({
  recommendations,
  isRecommending,
  onRequestRecommendations,
  skillsLength,
}) => {
  return (
    <div className="flex-1 rounded-2xl border border-blue-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col sm:flex-row flex-wrap items-start sm:items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-cyan-100 p-1.5">
            <svg
              className="h-4 w-4 text-cyan-600"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2-2v2m8 0V6a2 2 0 012 2v6a2 2 0 01-2 2H8a2 2 0 01-2-2V8a2 2 0 012-2V6"
              />
            </svg>
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900">
              Recommendations
            </h3>
            <p className="text-sm text-slate-600">AI-curated matches</p>
          </div>
        </div>
        <button
          onClick={onRequestRecommendations}
          disabled={isRecommending || !skillsLength}
          className="inline-flex items-center justify-center gap-1 rounded-xl border border-blue-600 bg-blue-600 px-4 py-2.5 text-sm font-bold text-white transition-all hover:bg-blue-700 disabled:border-slate-300 disabled:bg-slate-200 disabled:text-slate-400"
        >
          {isRecommending ? (
            <>
              <span className="h-3 w-3 animate-spin rounded-full border border-white/40 border-t-white" />
              Finding...
            </>
          ) : (
            <>
              <svg
                className="h-3 w-3"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 5l7 7-7 7"
                />
              </svg>
              Get Matches
            </>
          )}
        </button>
      </div>
      <div className="mt-4">
        <RecommendationGrid
          recommendations={recommendations}
          isLoading={isRecommending}
        />
      </div>
    </div>
  );
};

export default RecommendationsSection;
