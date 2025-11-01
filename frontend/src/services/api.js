import { API_ENDPOINTS } from '../constants/api.js';
import { buildFormData } from '../utils/helpers.js';

export const uploadResume = async (file) => {
  const formData = buildFormData(file);
  const response = await fetch(API_ENDPOINTS.UPLOAD, {
    method: "POST",
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error("Failed to upload resume");
  }
  
  return response.json();
};

export const getRecommendations = async (skills, topK = 6) => {
  const response = await fetch(API_ENDPOINTS.RECOMMEND, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ skills, top_k: topK }),
  });
  
  if (!response.ok) {
    throw new Error("Failed to fetch recommendations");
  }
  
  return response.json();
};