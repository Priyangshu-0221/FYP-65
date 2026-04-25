"use client";

import {
  ResumeUploadResponse,
  RecommendationRequest,
  RecommendationResponse,
  Internship,
  ErrorResponse,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
const ENDPOINTS = {
  HEALTH: `${API_BASE}/health`,
  UPLOAD: `${API_BASE}/upload`,
  RECOMMEND: `${API_BASE}/recommend`,
};

class APIError extends Error {
  constructor(
    public status: number,
    message: string,
    public data?: unknown,
  ) {
    super(message);
    this.name = "APIError";
  }
}

/**
 * Upload a resume file and extract skills
 */
export async function uploadResume(file: File): Promise<ResumeUploadResponse> {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(ENDPOINTS.UPLOAD, {
      method: "POST",
      body: formData,
      headers: {
        // Omit Content-Type to let browser set it with boundary
      },
    });

    if (!response.ok) {
      const errorData: ErrorResponse = await response.json().catch(() => ({
        error: "Unknown error",
      }));
      throw new APIError(
        response.status,
        errorData.detail || errorData.error || "Failed to upload resume",
        errorData,
      );
    }

    const data = await response.json();
    return data as ResumeUploadResponse;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      500,
      `Upload failed: ${error instanceof Error ? error.message : "Unknown error"}`,
    );
  }
}

/**
 * Get internship recommendations based on skills and marks
 */
export async function getRecommendations(
  skills: string[],
  marks?: { cgpa?: number; percentage?: number },
  topK: number = 6,
): Promise<RecommendationResponse> {
  try {
    const payload: RecommendationRequest = {
      skills: skills.map((s) => s.toLowerCase()),
      top_k: topK,
      ...(marks && { marks }),
      skill_count: skills.length,
    };

    const response = await fetch(ENDPOINTS.RECOMMEND, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData: ErrorResponse = await response.json().catch(() => ({
        error: "Unknown error",
      }));
      throw new APIError(
        response.status,
        errorData.detail || errorData.error || "Failed to get recommendations",
        errorData,
      );
    }

    const data = await response.json();
    return {
      recommendations: data.recommendations || [],
      recommended_skills: data.recommended_skills || [],
    };
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      500,
      `Recommendation request failed: ${error instanceof Error ? error.message : "Unknown error"}`,
    );
  }
}

/**
 * Check API health
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(ENDPOINTS.HEALTH, {
      method: "GET",
    });
    return response.ok;
  } catch {
    return false;
  }
}

export { APIError, ENDPOINTS };
