import { useState } from 'react';
import { uploadResume as uploadResumeApi, getRecommendations as getRecommendationsApi } from '../services/api.js';
import { toast } from '../utils/helpers.js';

export const useResumeUpload = () => {
  const [file, setFile] = useState(null);
  const [skills, setSkills] = useState([]);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setSkills([]);
  };

  const uploadResume = async () => {
    if (!file) {
      toast({
        title: "Upload a resume",
        description: "Please select a PDF or text resume first.",
        status: "warning",
        duration: 4000,
        isClosable: true,
      });
      return;
    }

    setIsUploading(true);
    try {
      const data = await uploadResumeApi(file);
      setSkills(data.skills ?? []);
      toast({
        title: data.skills?.length
          ? "Skills extracted successfully."
          : "Upload complete.",
        description: data.skills?.length
          ? `Found ${data.skills.length} skills from your resume.`
          : "Your resume has been processed.",
        status: "success",
        duration: 4000,
        isClosable: true,
      });
    } catch (error) {
      console.error(error);
      toast({
        title: "Upload failed",
        description: "Please try again later.",
        status: "error",
        duration: 4000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
    }
  };

  return {
    file,
    skills,
    isUploading,
    handleFileChange,
    uploadResume,
    setSkills,
  };
};