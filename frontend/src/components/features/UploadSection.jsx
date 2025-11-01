import React from "react";

const UploadSection = ({ file, isUploading, onFileChange, onUpload }) => {
  return (
    <div className="group rounded-2xl border-2 border-blue-200/50 bg-linear-to-br from-blue-50 via-white to-cyan-50 p-4 shadow-lg transition-all hover:border-blue-300/60">
      <div className="flex items-center gap-3">
        <div className="rounded-xl bg-blue-600 p-2">
          <svg
            className="h-5 w-5 text-white"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        <div>
          <h3 className="text-lg sm:text-xl font-bold text-slate-900">
            Upload Resume
          </h3>
          <p className="text-sm sm:text-base text-slate-600">
            AI-powered skill extraction
          </p>
        </div>
      </div>

      <label className="mt-4 block cursor-pointer rounded-xl border-2 border-dashed border-blue-300 bg-white/60 p-3 sm:p-4 text-center transition-all hover:border-blue-400 hover:bg-blue-50/50">
        <div className="space-y-2">
          <div className="mx-auto w-fit rounded-full bg-blue-100 p-2">
            <svg
              className="h-5 w-5 text-blue-600"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 4v16m8-8H4"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm sm:text-base font-semibold text-slate-700">
              Choose resume file
            </p>
            <p className="text-xs sm:text-sm text-slate-500">Please Upload your resume in .pdf format only</p>
          </div>
        </div>
        <input
          className="mt-3 block w-full cursor-pointer rounded-lg border border-blue-200 bg-white px-3 py-2 text-xs sm:text-sm text-slate-600 file:mr-3 file:rounded file:border-0 file:bg-linear-to-r file:from-blue-600 file:to-cyan-600 file:px-2 sm:file:px-3 file:py-1 file:text-xs sm:file:text-sm file:font-semibold file:text-white hover:file:brightness-105"
          type="file"
          accept=".pdf,.txt"
          onChange={onFileChange}
        />
      </label>

      <button
        onClick={onUpload}
        disabled={!file || isUploading}
        className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-linear-to-r from-blue-600 via-cyan-600 to-blue-700 px-4 py-2.5 sm:py-3 text-sm sm:text-base font-bold text-white shadow-lg transition-all hover:-translate-y-0.5 disabled:translate-y-0 disabled:opacity-50"
      >
        {isUploading ? (
          <>
            <span className="h-3 w-3 animate-spin rounded-full border border-white/40 border-t-white" />
            Analyzing...
          </>
        ) : (
          <>
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
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            Extract Skills
          </>
        )}
      </button>
    </div>
  );
};

export default UploadSection;
