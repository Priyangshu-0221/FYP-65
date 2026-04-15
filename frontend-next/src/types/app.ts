// Application State Types

export interface UploadState {
  file: File | null;
  skills: string[];
  isUploading: boolean;
  error: string | null;
}

export interface AcademicMarks {
  cgpa?: number;
  percentage?: number;
}

export interface RecommendationState {
  recommendations: any[];
  isLoading: boolean;
  error: string | null;
}

export interface UserSession {
  email: string;
  name?: string;
  image?: string;
}
