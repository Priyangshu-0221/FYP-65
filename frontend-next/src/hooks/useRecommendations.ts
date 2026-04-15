"use client";

import { useState } from "react";
import { getRecommendations, APIError } from "@/services/api";
import type { AcademicMarks } from "@/types/app";
import type { Internship } from "@/types/api";

export function useRecommendations() {
  const [recommendations, setRecommendations] = useState<Internship[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestRecommendations = async (
    skills: string[],
    marks?: AcademicMarks,
  ) => {
    if (!skills || skills.length === 0) {
      setError("Please provide at least one skill");
      return [];
    }

    setIsLoading(true);
    setError(null);
    try {
      const results = await getRecommendations(skills, marks);
      setRecommendations(results);
      return results;
    } catch (err) {
      const errorMessage =
        err instanceof APIError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Unknown error occurred";
      setError(errorMessage);
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
