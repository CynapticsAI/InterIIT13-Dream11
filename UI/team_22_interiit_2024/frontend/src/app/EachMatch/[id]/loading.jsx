import React from 'react'
import { Skeleton } from "@/components/ui/skeleton"
import { Card } from "@/components/ui/card"

export default function EachMatchCardSkeleton() {
  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header Skeleton */}
      <div className="bg-gradient-to-r from-red-900 via-black to-gray-900 p-4 text-white">
        <div className="max-w-7xl mx-auto flex items-center gap-4">
          <Skeleton className="w-10 h-10 rounded-full" />
          <div className="flex items-center justify-center gap-2 w-full">
            <div className="flex items-center gap-2">
              <Skeleton className="w-8 h-8" />
              <Skeleton className="w-32 h-6" />
            </div>
            <Skeleton className="w-8 h-6 mx-2" />
            <div className="flex items-center gap-2">
              <Skeleton className="w-32 h-6" />
              <Skeleton className="w-8 h-8" />
            </div>
          </div>
        </div>
        <Skeleton className="w-24 h-4 mx-auto mt-2" />
      </div>

      {/* Main Content Skeleton */}
      <div className="max-w-7xl mx-auto p-4 md:p-8 space-y-8">
        <Skeleton className="w-64 h-8 mx-auto" />
        
        {/* Create Team Card Skeleton */}
        <Card className="overflow-hidden max-w-2xl mx-auto">
          <div className="relative">
            <Skeleton className="w-full h-52" />
            <Skeleton className="w-full h-16" />
          </div>
        </Card>

        {/* Divider Skeleton */}
        <div className="flex items-center gap-4 max-w-2xl mx-auto">
          <div className="h-px bg-gray-300 flex-1" />
          <Skeleton className="w-8 h-8 rounded-full" />
          <div className="h-px bg-gray-300 flex-1" />
        </div>

        <Skeleton className="w-72 h-8 mx-auto" />

        {/* Expert Cards Container Skeleton */}
        <div className="w-1/2 flex justify-center items-center mx-auto">
          {/* Expert Card Skeleton */}
          <Card className="p-4 w-full">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <Skeleton className="w-12 h-12 rounded-full" />
                <Skeleton className="w-32 h-6" />
              </div>
              <div className="flex items-center gap-2">
                <Skeleton className="w-6 h-6" />
                <Skeleton className="w-8 h-6" />
              </div>
            </div>

            <Skeleton className="w-full h-48 rounded-lg" />

            <div className="flex flex-col sm:flex-row items-center justify-between mt-4 gap-4">
              <Skeleton className="w-32 h-6" />
              <Skeleton className="w-full sm:w-40 h-10" />
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

