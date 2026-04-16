"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BriefcaseBusiness,
  ExternalLink,
  MapPin,
  Sparkles,
} from "lucide-react";
import type { Internship } from "@/types/api";

interface RecommendationsGridProps {
  recommendations: Internship[];
  isLoading?: boolean;
}

export function RecommendationsGrid({
  recommendations,
  isLoading = false,
}: RecommendationsGridProps) {
  if (isLoading) {
    return (
      <Card className="app-surface border-[#d4d4d4] p-4 sm:p-5 md:p-6">
        <h2 className="mb-6 flex items-center gap-2 text-2xl font-bold text-[#111111]">
          <Sparkles className="h-6 w-6" />
          Recommended Internships
        </h2>
        <div className="grid grid-cols-1 gap-3 sm:gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="p-4">
              <Skeleton className="h-6 w-3/4 mb-2" />
              <Skeleton className="h-4 w-1/2 mb-2" />
              <Skeleton className="h-4 w-2/3" />
            </Card>
          ))}
        </div>
      </Card>
    );
  }

  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <Card className="app-surface border-[#d4d4d4] p-4 sm:p-5 md:p-6">
      <div className="space-y-4 sm:space-y-6">
        <div>
          <h2 className="mb-2 flex items-center gap-2 text-2xl font-bold text-[#111111]">
            <Sparkles className="h-6 w-6" />
            Recommended Internships ({recommendations.length})
          </h2>
          <p className="text-sm text-[#4a4a4a]">
            Based on your skills and academic profile
          </p>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:gap-4 md:grid-cols-2 lg:grid-cols-3">
          {recommendations.map((internship) => (
            <Card
              key={internship.id}
              className="cursor-pointer border border-[#d4d4d4] bg-white p-3 sm:p-4 transition-all hover:-translate-y-0.5 hover:shadow-lg"
            >
              <div className="space-y-3">
                {/* Header */}
                <div>
                  <h3 className="line-clamp-2 font-bold text-[#111111]">
                    {internship.title}
                  </h3>
                  <p className="mt-1 inline-flex items-center gap-2 text-sm font-medium text-[#2f2f2f]">
                    <BriefcaseBusiness className="h-4 w-4 text-[#111111]" />
                    {internship.company}
                  </p>
                </div>

                {/* Location & Category */}
                <div className="flex gap-2 flex-wrap">
                  <Badge
                    variant="outline"
                    className="gap-1 border-[#bcbcbc] text-xs text-[#2f2f2f]"
                  >
                    <MapPin className="h-3 w-3" />
                    {internship.location}
                  </Badge>
                  <Badge className="bg-black text-xs text-white hover:bg-[#2b2b2b]">
                    {internship.category}
                  </Badge>
                </div>

                {/* Skills */}
                <div>
                  <p className="mb-1 text-xs font-medium text-[#4a4a4a]">
                    Required Skills:
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {internship.skills.slice(0, 4).map((skill) => (
                      <Badge
                        key={skill}
                        variant="secondary"
                        className="border border-[#cfcfcf] bg-[#f1f1f1] text-xs text-[#1f1f1f]"
                      >
                        {skill}
                      </Badge>
                    ))}
                    {internship.skills.length > 4 && (
                      <Badge
                        variant="secondary"
                        className="border border-[#cfcfcf] bg-[#f1f1f1] text-xs text-[#1f1f1f]"
                      >
                        +{internship.skills.length - 4}
                      </Badge>
                    )}
                  </div>
                </div>

                {/* Description */}
                {internship.description && (
                  <p className="line-clamp-2 text-sm text-[#4a4a4a]">
                    {internship.description}
                  </p>
                )}

                {/* Apply Button */}
                <Button asChild className="w-full">
                  <a
                    href={internship.apply_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center gap-2"
                  >
                    View & Apply
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </Button>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </Card>
  );
}
