"use client";

import { useState } from "react";
import { getRecommendations, APIError } from "@/services/api";
import type { AcademicMarks } from "@/types/app";
import type { Internship } from "@/types/api";
import { toast } from "react-toastify";

export function useRecommendations() {
  const [recommendations, setRecommendations] = useState<Internship[]>([]);
  const [recommendedSkills, setRecommendedSkills] = useState<string[]>([]);
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
      return { recommendations: [], recommended_skills: [] };
    }

    setIsLoading(true);
    setError(null);
    try {
      const results = await getRecommendations(skills, marks);
      setRecommendations(results.recommendations);
      setRecommendedSkills(results.recommended_skills);
      toast.success(`Found ${results.recommendations.length} internship matches`);
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
      return { recommendations: [], recommended_skills: [] };
    } finally {
      setIsLoading(false);
    }
  };

  const resetRecommendations = () => {
    setRecommendations([]);
    setRecommendedSkills([]);
    setError(null);
  };

  return {
    recommendations,
    recommendedSkills,
    isLoading,
    error,
    requestRecommendations,
    resetRecommendations,
    setRecommendations,
  };
}
