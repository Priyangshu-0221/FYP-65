"use client";

import React, { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FileCheck2, FileText, Loader2, UploadCloud } from "lucide-react";
import { toast } from "sonner";

interface UploadSectionProps {
  onUpload: (file: File) => Promise<unknown>;
  isUploading: boolean;
  fileName?: string;
}

export function UploadSection({
  onUpload,
  isUploading,
  fileName,
}: UploadSectionProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const validateFile = (file: File): boolean => {
    if (!file.type.includes("pdf")) {
      toast.error("Please upload a PDF file");
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File size must be less than 10MB");
      return false;
    }
    return true;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      const file = files[0];
      if (validateFile(file)) {
        handleFileUpload(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (validateFile(file)) {
        handleFileUpload(file);
      }
    }
  };

  const handleFileUpload = async (file: File) => {
    try {
      await onUpload(file);
      toast.success("Resume uploaded successfully!");
    } catch (error) {
      console.error("Upload error:", error);
      // Error toast is shown by parent component
    }
  };

  return (
    <Card className="app-surface border-[#d8e0ed] p-8">
      <div className="space-y-4">
        <div>
          <h2 className="mb-2 flex items-center gap-2 text-2xl font-bold text-[#1d3b72]">
            <FileCheck2 className="h-6 w-6" />
            Upload Your Resume
          </h2>
          <p className="text-[#5a687d]">
            Upload a PDF resume to extract skills and get personalized
            recommendations
          </p>
        </div>

        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
            dragActive
              ? "border-[#27549f] bg-[#eef3fb]"
              : "border-[#c8d4e9] bg-[#fbfcff] hover:border-[#27549f]"
          }`}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            disabled={isUploading}
            className="hidden"
          />

          <div className="space-y-2">
            <p className="mx-auto inline-flex h-12 w-12 items-center justify-center rounded-full border border-[#c8d4e9] bg-white text-[#1d3b72]">
              <FileText className="h-5 w-5" />
            </p>
            <p className="font-semibold text-[#22314f]">
              {fileName || "Drag and drop your PDF here"}
            </p>
            <p className="text-sm text-[#5a687d]">or click to browse files</p>
            <p className="text-xs text-[#7484a0]">PDF up to 10MB</p>
          </div>
        </div>

        <Button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="w-full gap-2"
        >
          {isUploading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Uploading...
            </>
          ) : (
            <>
              <UploadCloud className="h-4 w-4" />
              Select Resume
            </>
          )}
        </Button>
      </div>
    </Card>
  );
}
