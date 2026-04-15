"use client";

import { useState } from "react";
import { uploadResume, APIError } from "@/services/api";

export function useResumeUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [skills, setSkills] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (newFile: File) => {
    setFile(newFile);
    setError(null);
  };

  const handleUpload = async (selectedFile: File) => {
    setIsUploading(true);
    setError(null);
    try {
      const response = await uploadResume(selectedFile);
      setSkills(response.skills);
      setFile(selectedFile);
      return response;
    } catch (err) {
      const errorMessage =
        err instanceof APIError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Unknown error occurred";
      setError(errorMessage);
      throw err;
    } finally {
      setIsUploading(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setSkills([]);
    setError(null);
  };

  return {
    file,
    skills,
    isUploading,
    error,
    handleFileChange,
    handleUpload,
    resetUpload,
    setSkills,
  };
}
