"use client";

import { useState } from "react";
import { getRecommendations, APIError } from "@/services/api";
import type { AcademicMarks } from "@/types/app";
import type { Internship } from "@/types/api";
import { toast } from "react-toastify";

export function useRecommendations() {
  const [recommendations, setRecommendations] = useState<Internship[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestRecommendations = async (
    skills: string[],
    marks?: AcademicMarks,
  ) => {
    if (!skills || skills.length === 0) {
      const message = "Please provide at least one skill";
      setError(message);
      toast.error(message);
      return [];
    }

    setIsLoading(true);
    setError(null);
    try {
      const results = await getRecommendations(skills, marks);
      setRecommendations(results);
      toast.success(`Found ${results.length} internship matches`);
      return results;
    } catch (err) {
      const errorMessage =
        err instanceof APIError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Unknown error occurred";
      setError(errorMessage);
      toast.error(errorMessage);
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  const resetRecommendations = () => {
    setRecommendations([]);
    setError(null);
  };

  return {
    recommendations,
    isLoading,
    error,
    requestRecommendations,
    resetRecommendations,
    setRecommendations,
  };
}
