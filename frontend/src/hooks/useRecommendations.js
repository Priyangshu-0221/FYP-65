import { useState } from 'react';
import { getRecommendations as getRecommendationsApi } from '../services/api.js';
import { toast } from '../utils/helpers.js';

export const useRecommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [isRecommending, setIsRecommending] = useState(false);

  const requestRecommendations = async (skills) => {
    if (!skills.length) {
      toast({
        title: "No skills available",
        description: "Upload a resume first or enter skills manually.",
        status: "info",
        duration: 4000,
        isClosable: true,
      });
      return;
    }

    setIsRecommending(true);
    try {
      const data = await getRecommendationsApi(skills, 6);
      setRecommendations(data.recommendations ?? []);
    } catch (error) {
      console.error(error);
      toast({
        title: "Recommendation failed",
        description: "Please try again later.",
        status: "error",
        duration: 4000,
        isClosable: true,
      });
    } finally {
      setIsRecommending(false);
    }
  };

  return {
    recommendations,
    isRecommending,
    requestRecommendations,
    setRecommendations,
  };
};