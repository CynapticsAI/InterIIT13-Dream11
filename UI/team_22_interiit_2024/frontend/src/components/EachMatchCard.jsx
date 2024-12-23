"use client";

import React from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, User, Award, Flag, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import teampagephoto from "../../public/images/team_page_photo.jpg";
import Image from "next/image";

export default function EachMatchCard({ match, id }) {
  const router = useRouter();
  return (
    <div>
      <div className="min-h-screen bg-gray-100">
        {/* Header */}
        <div className="bg-gradient-to-r from-red-900 via-black to-gray-900 p-4 text-white">
          <div className="max-w-7xl mx-auto flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="text-white "
              onClick={() => router.back()}
              aria-label="Go back"
            >
              <ArrowLeft className="w-6 h-6" />
            </Button>
            <div className="flex items-center justify-center gap-2 w-full">
              <div className="flex items-center gap-2">
                <Flag className="w-8 h-8" />
                <span className="text-xl font-semibold">
                  {match ? match[0].team_a : "Team A"}
                </span>
              </div>
              <span className="text-gray-400 mx-2">Vs</span>
              <div className="flex items-center gap-2">
                <span className="text-xl font-semibold">
                  {match ? match[0].team_b : "Team B"}
                </span>
                <Flag className="w-8 h-8" />
              </div>
            </div>
          </div>
          <div className="text-center mt-2 text-gray-300">44h left</div>
        </div>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto p-4 md:p-8 space-y-8">
          <h1 className="text-2xl md:text-3xl font-bold text-center">
            Pick your favourite players
          </h1>

          {/* Create Team Card */}
          <Card className="overflow-hidden max-w-2xl mx-auto">
            <div className="relative">
              <div className="bg-gray-200 h-52 flex items-center justify-center">
                <Image
                  src={teampagephoto}
                  alt="South Africa"
                  className=""
                  placeholder="blur"
                />
              </div>
              <Link
                href={`/EachMatch/${id}/SelectTeam`}
                className="h-fit w-fit"
              >
                <Button className="w-full bg-green-600 hover:bg-green-700 text-white py-4 md:py-6 rounded-none text-lg md:text-xl">
                  CREATE YOUR OWN TEAM
                </Button>
              </Link>
            </div>
          </Card>

          {/* Divider */}
          <div className="flex items-center gap-4 max-w-2xl mx-auto">
            <div className="h-px bg-gray-300 flex-1" />
            <div className="text-xl font-bold">OR</div>
            <div className="h-px bg-gray-300 flex-1" />
          </div>

          <h2 className="text-2xl md:text-3xl font-bold text-center">
            Pick a Team Made by our AI
          </h2>

          {/* Expert Cards Container */}
          <div className="w-1/2 flex justify-center items-center mx-auto">
            {/* Expert Card */}
            <Card className="p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <User className="w-12 h-12 text-gray-400" />
                  <span className="text-xl font-semibold">Dream 11 AI</span>
                </div>
                <div className="flex items-center gap-2">
                  <Award className="w-6 h-6 text-yellow-500" />
                  <span className="text-xl font-semibold">10</span>
                </div>
              </div>

              <div className="bg-green-600 rounded-lg p-4 text-white">
                <div className="text-sm mb-2">Mega</div>
                <div className="flex justify-between mb-4">
                  <div>
                    <div>{match[0]?.team_a?.name}</div>
                  </div>
                  <div>
                    <div>{match[0]?.team_b?.name}</div>
                  </div>
                </div>
                <div className="flex justify-between">
                  <div className="flex items-center gap-2">
                    <User className="w-12 h-12 bg-white text-green-600 rounded-full p-2" />
                    <div>
                      <div className="text-sm">S Samson</div>
                      <div className="bg-black rounded-full w-6 h-6 flex items-center justify-center text-sm">
                        C
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <User className="w-12 h-12 bg-white text-green-600 rounded-full p-2" />
                    <div>
                      <div className="text-sm">H Pandya</div>
                      <div className="bg-black rounded-full w-6 h-6 flex items-center justify-center text-sm">
                        VC
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row items-center justify-between mt-4 gap-4">
                <div className="text-amber-800 font-semibold">
                  8.7K times picked
                </div>
                <Button
                  variant="outline"
                  className="gap-2 w-full sm:w-auto"
                  onClick={() =>
                    router.push(`/EachMatch/${id}/SelectTeam/optimal`)
                  }
                >
                  <Users className="w-5 h-5" />
                  PICK THIS TEAM
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
