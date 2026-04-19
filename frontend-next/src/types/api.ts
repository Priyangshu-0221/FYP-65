// API Request & Response Types

export interface ResumeUploadResponse {
  success: boolean;
  text: string;
  skills: string[];
  category: string;
  education: string;
  experience: string;
  skill_count: number;
  message?: string;
  cgpa?: number;
  percentage?: number;
}

export interface RecommendationRequest {
  skills: string[];
  top_k?: number;
  marks?: {
    cgpa?: number;
    percentage?: number;
  };
  skill_count?: number;
}

export interface Internship {
  id: string;
  title: string;
  company: string;
  location: string;
  category: string;
  skills: string[];
  description: string;
  apply_link: string;
}

export interface RecommendationResponse {
  recommendations: Internship[];
  recommended_skills: string[];
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}

// Type guards
export function isResumeUploadResponse(
  data: unknown,
): data is ResumeUploadResponse {
  const obj = data as Record<string, unknown>;
  return (
    typeof obj === "object" &&
    typeof obj.success === "boolean" &&
    typeof obj.text === "string" &&
    Array.isArray(obj.skills) &&
    typeof obj.skill_count === "number"
  );
}

export function isInternship(data: unknown): data is Internship {
  const obj = data as Record<string, unknown>;
  return (
    typeof obj === "object" &&
    typeof obj.id === "string" &&
    typeof obj.title === "string" &&
    typeof obj.company === "string" &&
    Array.isArray(obj.skills)
  );
}
