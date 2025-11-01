import React from "react";
import { Sidebar, Dashboard } from "./components/index.js";
import { useResumeUpload, useRecommendations } from "./hooks/index.js";

function App() {
  // Custom hooks for state management
  const { file, skills, isUploading, handleFileChange, uploadResume } =
    useResumeUpload();

  const { recommendations, isRecommending, requestRecommendations } =
    useRecommendations();

  // Handle recommendations request
  const handleRequestRecommendations = () => {
    requestRecommendations(skills);
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-900 via-gray-900 to-slate-800 text-white">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col lg:flex-row">
        <Sidebar />
        <Dashboard
          file={file}
          skills={skills}
          recommendations={recommendations}
          isUploading={isUploading}
          isRecommending={isRecommending}
          onFileChange={handleFileChange}
          onUpload={uploadResume}
          onRequestRecommendations={handleRequestRecommendations}
        />
      </div>
    </div>
  );
}

export default App;
