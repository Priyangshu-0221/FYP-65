import React from "react";

const SkillsList = ({ skills }) => {
  if (!skills || !skills.length) {
    return <p className="text-sm text-slate-500">No skills extracted yet.</p>;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {skills.map((skill) => (
        <span
          key={skill}
          className="inline-flex items-center gap-1.5 rounded-full border border-blue-200/80 bg-blue-50 px-3 py-1.5 text-sm font-semibold text-blue-700 shadow-sm transition-all hover:bg-blue-100"
        >
          <span className="h-2 w-2 rounded-full bg-blue-400" />
          {skill}
        </span>
      ))}
    </div>
  );
};

export default SkillsList;
