import React from "react";

const RecommendationGrid = ({ recommendations, isLoading }) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-3">
        {Array.from({ length: 2 }).map((_, i) => (
          <div
            key={i}
            className="h-20 animate-pulse rounded-xl border border-blue-200/60 bg-linear-to-br from-blue-50/70 via-white to-cyan-50"
          />
        ))}
      </div>
    );
  }

  if (!recommendations || !recommendations.length) {
    return <p className="text-sm text-slate-500">No recommendations yet.</p>;
  }

  return (
    <div className="grid grid-cols-1 gap-3">
      {recommendations.slice(0, 3).map((internship) => (
        <article
          key={internship.id}
          className="group relative overflow-hidden rounded-xl border border-blue-200/70 bg-white/95 p-3 shadow-sm transition-all duration-200 hover:border-blue-400/80 hover:shadow-md"
        >
          <span className="absolute inset-x-0 top-0 h-0.5 bg-linear-to-r from-blue-500 via-cyan-500 to-blue-500" />
          <div className="relative z-10">
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1">
                <h3 className="text-base font-bold text-slate-900 group-hover:text-blue-600 transition-colors line-clamp-1">
                  {internship.title}
                </h3>
                <p className="text-sm text-slate-600 font-medium">
                  {internship.company} • {internship.location}
                </p>
              </div>
              <span className="rounded-full border border-blue-200/60 bg-blue-100 px-2.5 py-1 text-sm font-bold text-blue-600 shrink-0">
                {internship.category}
              </span>
            </div>
            <p className="text-sm text-slate-600 leading-relaxed mt-2 line-clamp-2">
              {internship.description}
            </p>
            <div className="flex items-center justify-between mt-3">
              <div className="flex flex-wrap gap-1">
                {internship.skills.slice(0, 3).map((skill) => (
                  <span
                    key={skill}
                    className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-1 text-sm font-medium text-slate-600 border border-slate-200"
                  >
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
                    {skill}
                  </span>
                ))}
                {internship.skills.length > 3 && (
                  <span className="text-sm text-slate-500">
                    +{internship.skills.length - 3} more
                  </span>
                )}
              </div>
              <a
                className="inline-flex items-center gap-1 text-sm font-bold text-blue-600 transition-colors hover:text-blue-700 shrink-0"
                href={internship.apply_link}
                target="_blank"
                rel="noreferrer"
              >
                Apply
                <svg
                  className="h-4 w-4"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                  />
                </svg>
              </a>
            </div>
          </div>
        </article>
      ))}
      {recommendations.length > 3 && (
        <p className="text-xs text-center text-slate-500 mt-2">
          Showing 3 of {recommendations.length} recommendations
        </p>
      )}
    </div>
  );
};

export default RecommendationGrid;
