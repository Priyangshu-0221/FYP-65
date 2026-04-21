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
      <Card className="border-slate-200 p-4 sm:p-5 md:p-6">
        <h2 className="mb-6 flex items-center gap-2 text-2xl font-bold text-slate-900">
          <Sparkles className="h-6 w-6 text-sky-600" />
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
    <Card className="border-slate-200 p-4 sm:p-5 md:p-6">
      <div className="space-y-4 sm:space-y-6">
        <div>
          <h2 className="mb-2 flex items-center gap-2 text-2xl font-bold text-slate-900">
            <Sparkles className="h-6 w-6 text-sky-600" />
            Recommended Internships ({recommendations.length})
          </h2>
          <p className="text-sm text-slate-600">
            Based on your skills and academic profile
          </p>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:gap-4 md:grid-cols-2 lg:grid-cols-3">
          {recommendations.map((internship) => (
            <Card
              key={internship.id}
              className="cursor-pointer border border-slate-200 bg-slate-50 p-3 transition-all hover:-translate-y-0.5 hover:border-sky-200 hover:bg-sky-50 sm:p-4"
            >
              <div className="space-y-3">
                {/* Header */}
                <div>
                  <h3 className="line-clamp-2 font-bold text-slate-900">
                    {internship.title}
                  </h3>
                  <p className="mt-1 inline-flex items-center gap-2 text-sm font-medium text-slate-600">
                    <BriefcaseBusiness className="h-4 w-4 text-emerald-600" />
                    {internship.company}
                  </p>
                </div>

                {/* Location & Category */}
                <div className="flex gap-2 flex-wrap">
                  <Badge
                    variant="outline"
                    className="gap-1 border-slate-200 bg-white text-xs text-slate-600"
                  >
                    <MapPin className="h-3 w-3" />
                    {internship.location}
                  </Badge>
                  <Badge className="bg-emerald-600 text-xs text-white hover:bg-emerald-500">
                    {internship.category}
                  </Badge>
                </div>

                {/* Skills */}
                <div>
                  <p className="mb-1 text-xs font-medium text-slate-500">
                    Required Skills:
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {internship.skills.slice(0, 4).map((skill) => (
                      <Badge
                        key={skill}
                        variant="secondary"
                        className="border border-emerald-200 bg-emerald-50 text-xs text-emerald-700"
                      >
                        {skill}
                      </Badge>
                    ))}
                    {internship.skills.length > 4 && (
                      <Badge
                        variant="secondary"
                        className="border border-emerald-200 bg-emerald-50 text-xs text-emerald-700"
                      >
                        +{internship.skills.length - 4}
                      </Badge>
                    )}
                  </div>
                </div>

                {/* Description */}
                {internship.description && (
                  <p className="line-clamp-2 text-sm text-slate-600">
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
