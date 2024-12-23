"use client";

import React, { useEffect, useState } from "react";
import {
  ArrowLeft,
  Info,
  Minus,
  ChevronRight,
  Trophy,
  ChevronDown,
  ChevronUp,
  Loader2,
  User,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { set } from "date-fns";

const PlayerCard = ({
  player,
  isSelected,
  onToggle,
  isPredicted = false,
  current_year,
  match_type,
}) => {
  const currentYear = current_year || new Date().getFullYear();

  return (
    <div
      className={`flex items-center p-3 sm:p-4 border-b transition-colors ${
        isSelected ? "bg-green-100" : ""
      } ${isPredicted ? "border-l-4 border-l-yellow-500" : ""}`}
    >
      <Info className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400 mr-2 sm:mr-3 flex-shrink-0" />
      <div className="flex flex-1 items-center">
        <img
          src={`https://ui-avatars.com/api/?name=${player?.name}&background=random`}
          alt={player.name}
          className="w-12 h-12 sm:w-20 sm:h-20 rounded-lg object-cover mr-3 sm:mr-4"
        />
        <div className="flex-1 min-w-0">
          <h3 className="text-base sm:text-lg font-semibold truncate">
            {player?.name}
          </h3>
          <p className="text-sm text-gray-600">{player?.role}</p>
          <div className="flex items-center mt-1">
            <div className="w-2 h-2 bg-blue-600 rounded-full mr-2" />
            <span className="text-xs sm:text-sm text-gray-600">
              Played last match
            </span>
          </div>
          <Link
            href={`/PlayerData/${player.name}/${player.identifier}`}
            className="text-blue-600 hover:underline text-sm mt-1 flex items-center"
          >
            <User className="w-4 h-4 mr-1" />
            View Profile
          </Link>
        </div>
        <div className="flex items-center gap-2 sm:gap-8 ml-2 sm:ml-4">
          <span className="text-lg sm:text-2xl font-semibold">
            {player.identifier}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 sm:w-10 sm:h-10 rounded-full border-2 border-green-500"
            onClick={(e) => {
              e.stopPropagation();
              onToggle(player.id);
            }}
          >
            {isSelected ? (
              <Minus className="w-4 h-4 sm:w-6 sm:h-6" />
            ) : (
              <span className="text-base sm:text-xl">+</span>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

const MatchDetailClient = ({ matchData, current_year, optimal = false }) => {
  const [selectedPlayers, setSelectedPlayers] = useState([]);
  const [activeTab, setActiveTab] = useState("batters");
  const [showPredictedPlayers, setShowPredictedPlayers] = useState(false);
  const [predictedPlayers, setPredictedPlayers] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isPredictedSectionOpen, setIsPredictedSectionOpen] = useState(true);

  const playersTeamA = matchData.team_a_players;
  const playersTeamB = matchData.team_b_players;
  const players = [...playersTeamA, ...playersTeamB];

  const setPrediction = async () => {
    const response = await fetch(
      "https://qwertyweb.xyz:8443/backend/api/v1/predict-players/",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          match_id: matchData.id,
          selected_players_id: selectedPlayers,
        }),
      }
    );

    if (!response.ok) {
      throw new Error("Failed to predict players");
    }

    const data = await response.json();
    setPredictedPlayers(data.predicted_players);
  };

  useEffect(() => {
    if (optimal) {
      setPrediction();
      setShowPredictedPlayers(true);
      handlePredictPlayers();
    }
  }, []);

  const togglePlayer = (playerId) => {
    setSelectedPlayers((prev) =>
      prev.includes(playerId)
        ? prev.filter((id) => id !== playerId)
        : [...prev, playerId]
    );
  };

  const selectedTeamCounts = {
    team_a: selectedPlayers.filter((playerId) =>
      playersTeamA.some((p) => p.id === playerId)
    ).length,
    team_b: selectedPlayers.filter((playerId) =>
      playersTeamB.some((p) => p.id === playerId)
    ).length,
  };

  const filteredPlayers = players.filter((player) => {
    switch (activeTab) {
      case "batters":
        return player.role === "BAT";
      case "bowlers":
        return player.role === "BOWL";
      case "allRounders":
        return player.role === "AR";
      case "wicketKeepers":
        return player.role === "WK";
      default:
        return true;
    }
  });

  const handlePredictPlayers = async () => {
    setIsLoading(true);
    setShowPredictedPlayers(true);
    setIsPredictedSectionOpen(true);
    try {
      setPrediction();
    } catch (error) {
      console.error("Failed to predict players:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUseThisTeam = () => {
    setSelectedPlayers(predictedPlayers.map((player) => player.id));
    setIsPredictedSectionOpen(false);
  };

  const router = useRouter();

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-gradient-to-b from-[#4a0e0e] to-[#1a1f25] text-white p-3 sm:p-4">
        <div className="flex items-center justify-between mb-3 sm:mb-4">
          <div className="flex items-center gap-2 sm:gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="text-white"
              onClick={() => router.back()}
              aria-label="Go back"
            >
              <ArrowLeft className="w-5 h-5 sm:w-6 sm:h-6" />
            </Button>
            <div>
              <h1 className="text-lg sm:text-xl font-bold">Create Team</h1>
              <p className="text-xs sm:text-sm text-gray-400">44h left</p>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <div className="bg-[#2a3038] px-2 sm:px-4 py-1 sm:py-2 rounded-lg flex items-center text-xs sm:text-sm">
              <Trophy className="w-3 h-3 sm:w-4 sm:h-4 text-yellow-500 mr-1" />
              <span className="text-yellow-500 font-bold">G</span>
              <span className="ml-1">GURUS</span>
            </div>
            <div className="bg-white text-black px-2 sm:px-4 py-1 sm:py-2 rounded-full font-bold text-xs sm:text-sm">
              PTS
            </div>
          </div>
        </div>

        <p className="text-center mb-3 sm:mb-4 text-xs sm:text-sm">
          Maximum of 9 players from one team
        </p>

        <div className="flex justify-center items-center gap-4 sm:gap-8 mb-3 sm:mb-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-green-600 flex items-center justify-center text-xs sm:text-sm">
              {matchData.team_a.slice(0, 3)}
            </div>
            <span className="text-xl sm:text-2xl font-bold">
              {selectedTeamCounts.team_a}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-blue-600 flex items-center justify-center text-xs sm:text-sm">
              {matchData.team_b.slice(0, 3)}
            </div>
            <span className="text-xl sm:text-2xl font-bold">
              {selectedTeamCounts.team_b}
            </span>
          </div>
        </div>

        <div className="flex gap-1 mb-3 sm:mb-4">
          {Array.from({ length: 11 }).map((_, index) => (
            <div
              key={index}
              className={`flex-1 h-1 sm:h-2 ${
                index < selectedPlayers.length ? "bg-green-500" : "bg-white"
              }`}
            />
          ))}
        </div>
      </header>

      <div className="bg-[#1a1f25] text-white p-3 sm:p-4 flex flex-wrap items-center justify-between text-xs sm:text-sm">
        <div className="flex items-center mb-2 sm:mb-0">
          <span className="font-bold mr-2">STATS</span>
          <ChevronRight className="w-3 h-3 sm:w-4 sm:h-4" />
        </div>
        <div className="flex flex-wrap items-center gap-3 sm:gap-6">
          <div className="flex items-center gap-1 sm:gap-2">
            <span className="text-gray-400">Pitch:</span>
            <span>Batting</span>
          </div>
          <div className="flex items-center gap-1 sm:gap-2">
            <span className="text-gray-400">Good for:</span>
            <span>Pace</span>
          </div>
          <div className="flex items-center gap-1 sm:gap-2">
            <span className="text-gray-400">Avg Score:</span>
            <span>201</span>
          </div>
        </div>
      </div>

      <div className="p-3 sm:p-4 bg-white">
        <Button
          onClick={handlePredictPlayers}
          className="w-full mb-3 sm:mb-4 bg-yellow-500 hover:bg-yellow-600 text-white"
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Predicting
              Players...
            </>
          ) : (
            "Predict Players"
          )}
        </Button>

        {showPredictedPlayers && (
          <div className="mb-4 sm:mb-6 border rounded-lg overflow-hidden">
            <div
              className="flex justify-between items-center bg-yellow-100 p-3 sm:p-4 cursor-pointer"
              onClick={() => setIsPredictedSectionOpen(!isPredictedSectionOpen)}
            >
              <h2 className="text-base sm:text-lg font-semibold">
                Predicted Players (11)
              </h2>
              <Button
                variant="ghost"
                size="sm"
                className="text-yellow-600 hover:text-yellow-700"
              >
                {isPredictedSectionOpen ? (
                  <ChevronUp className="w-4 h-4 sm:w-5 sm:h-5" />
                ) : (
                  <ChevronDown className="w-4 h-4 sm:w-5 sm:h-5" />
                )}
              </Button>
            </div>
            {isPredictedSectionOpen && (
              <>
                {predictedPlayers.map((player) => (
                  <PlayerCard
                    key={player.id}
                    player={player}
                    isSelected={selectedPlayers.includes(player.id)}
                    onToggle={togglePlayer}
                    isPredicted={true}
                    current_year={current_year}
                    match_type={matchData.match_type}
                  />
                ))}
                <div className="p-3 sm:p-4 bg-gray-50">
                  <Button
                    onClick={handleUseThisTeam}
                    className="w-full bg-green-500 hover:bg-green-600 text-white"
                  >
                    Use This Team
                  </Button>
                </div>
              </>
            )}
          </div>
        )}

        <div className="mb-3 sm:mb-4 flex flex-wrap gap-2">
          {["batters", "bowlers", "allRounders", "wicketKeepers"].map((tab) => (
            <button
              key={tab}
              className={`px-3 py-1 sm:px-4 sm:py-2 rounded-lg text-xs sm:text-sm ${
                activeTab === tab ? "bg-green-500 text-white" : "bg-gray-200"
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {filteredPlayers.map((player) => (
          <PlayerCard
            key={player.id}
            player={player}
            isSelected={selectedPlayers.includes(player.id)}
            onToggle={togglePlayer}
            current_year={current_year}
            match_type={matchData.match_type}
          />
        ))}
      </div>
    </div>
  );
};

export default MatchDetailClient;
