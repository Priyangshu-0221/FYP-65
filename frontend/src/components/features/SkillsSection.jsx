import React from "react";
import SkillsList from "../ui/SkillsList.jsx";

const SkillsSection = ({ skills }) => {
  return (
    <div className="rounded-2xl border border-blue-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-blue-100 p-1.5">
            <svg
              className="h-4 w-4 text-blue-600"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-bold text-slate-900">Detected Skills</h3>
        </div>
        <div className="rounded-full bg-blue-100 px-3 py-1">
          <span className="text-sm font-bold text-blue-600">
            {skills.length ? `${skills.length} found` : "Awaiting"}
          </span>
        </div>
      </div>
      <div className="mt-3">
        <SkillsList skills={skills} />
      </div>
    </div>
  );
};

export default SkillsSection;
