import React from "react";
import { ArrowLeft, Trophy, ChevronRight } from 'lucide-react';
import { Button } from "@/components/ui/button";

const SkeletonPlayerCard = () => (
  <div className="flex items-center p-4 border-b animate-pulse">
    <div className="w-5 h-5 bg-gray-200 rounded-full mr-3" />
    <div className="flex flex-1 items-center">
      <div className="w-20 h-20 bg-gray-200 rounded-lg mr-4" />
      <div className="flex-1">
        <div className="h-5 bg-gray-200 rounded w-3/4 mb-2" />
        <div className="h-4 bg-gray-200 rounded w-1/2 mb-2" />
        <div className="flex items-center mt-1">
          <div className="w-2 h-2 bg-gray-200 rounded-full mr-2" />
          <div className="h-3 bg-gray-200 rounded w-1/4" />
        </div>
        <div className="h-4 bg-gray-200 rounded w-1/3 mt-1" />
      </div>
      <div className="flex items-center gap-8 ml-4">
        <div className="h-6 w-6 bg-gray-200 rounded" />
        <div className="w-10 h-10 bg-gray-200 rounded-full" />
      </div>
    </div>
  </div>
);

const LoadingMatchDetail = () => {
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-gradient-to-b from-[#4a0e0e] to-[#1a1f25] text-white p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="text-white"
              aria-label="Go back"
            >
              <ArrowLeft className="w-6 h-6" />
            </Button>
            <div>
              <div className="h-6 bg-gray-200 rounded w-24 mb-1" />
              <div className="h-4 bg-gray-200 rounded w-16" />
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="bg-[#2a3038] px-4 py-2 rounded-lg flex items-center">
              <Trophy className="w-4 h-4 text-yellow-500 mr-1" />
              <div className="h-4 bg-gray-200 rounded w-16" />
            </div>
            <div className="bg-white px-4 py-2 rounded-full">
              <div className="h-4 bg-gray-200 rounded w-8" />
            </div>
          </div>
        </div>

        <div className="h-4 bg-gray-200 rounded w-3/4 mx-auto mb-4" />

        <div className="flex justify-center items-center gap-8 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-full bg-green-600" />
            <div className="h-8 bg-gray-200 rounded w-8" />
          </div>
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-full bg-blue-600" />
            <div className="h-8 bg-gray-200 rounded w-8" />
          </div>
        </div>

        <div className="flex gap-1 mb-4">
          {Array.from({ length: 11 }).map((_, index) => (
            <div key={index} className="flex-1 h-2 bg-white" />
          ))}
        </div>
      </header>

      <div className="bg-[#1a1f25] text-white p-4 flex flex-wrap items-center justify-between">
        <div className="flex items-center">
          <div className="h-4 bg-gray-200 rounded w-16 mr-2" />
          <ChevronRight className="w-4 h-4" />
        </div>
        <div className="flex flex-wrap items-center gap-6">
          {Array.from({ length: 3 }).map((_, index) => (
            <div key={index} className="flex items-center gap-2">
              <div className="h-4 bg-gray-200 rounded w-16" />
              <div className="h-4 bg-gray-200 rounded w-12" />
            </div>
          ))}
        </div>
      </div>

      <div className="p-4 bg-white">
        <div className="h-10 bg-gray-200 rounded w-full mb-4" />

        <div className="mb-4 flex flex-wrap gap-2">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="h-8 bg-gray-200 rounded w-24" />
          ))}
        </div>

        {Array.from({ length: 5 }).map((_, index) => (
          <SkeletonPlayerCard key={index} />
        ))}
      </div>
    </div>
  );
};

export default LoadingMatchDetail;

