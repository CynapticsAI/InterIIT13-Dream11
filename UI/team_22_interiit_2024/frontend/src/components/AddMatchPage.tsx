"use client";

import axios from "axios";
import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { format } from "date-fns";
import * as XLSX from "xlsx";
import { Loader, CalendarIcon, HelpCircle } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";

interface Player {
  id: number;
  name: string;
  role: string;
}

interface PlayersResponse {
  count: number;
  next: string | null;
  previous: string | null;
  results: Player[];
}

interface ExcelData {
  "Player Name": string;
  Squad: string;
  "Match Date": string;
  Format: string;
}

const MATCH_TYPES = ["ODI", "Test", "T20"] as const;
type MatchType = (typeof MATCH_TYPES)[number];

const PLAYERS_PER_PAGE = 1000;

export default function TeamSelectionForm() {
  const [allPlayers, setAllPlayers] = useState<Player[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [teamAPlayers, setTeamAPlayers] = useState<Player[]>([]);
  const [teamBPlayers, setTeamBPlayers] = useState<Player[]>([]);
  const [selectedDate, setSelectedDate] = useState<Date | undefined>();
  const [matchType, setMatchType] = useState<MatchType | "">("");
  const [teamAName, setTeamAName] = useState<string>("");
  const [teamBName, setTeamBName] = useState<string>("");
  const [isExcelMode, setIsExcelMode] = useState(false);
  const { toast } = useToast();

  const fetchPlayers = useCallback(
    async (page: number) => {
      setIsLoading(true);
      try {
        const response = await fetch(
          `https://qwertyweb.xyz:8443/backend/api/v1/players/?limit=${PLAYERS_PER_PAGE}&offset=${
            (page - 1) * PLAYERS_PER_PAGE
          }`
        );
        const data: PlayersResponse = await response.json();
        setAllPlayers((prevPlayers) => {
          const newPlayers = data.results.filter(
            (newPlayer) =>
              !prevPlayers.some(
                (existingPlayer) => existingPlayer.id === newPlayer.id
              )
          );
          return [...prevPlayers, ...newPlayers];
        });
        setTotalPages(Math.ceil(data.count / PLAYERS_PER_PAGE));
      } catch (error) {
        console.error("Error fetching players:", error);
        toast({
          variant: "destructive",
          title: "Error",
          description: "Error fetching players. Please try again.",
        });
      } finally {
        setIsLoading(false);
      }
    },
    [toast]
  );

  useEffect(() => {
    fetchPlayers(1);
  }, [fetchPlayers]);

  useEffect(() => {
    if (currentPage > Math.ceil(allPlayers.length / PLAYERS_PER_PAGE)) {
      fetchPlayers(currentPage);
    }
  }, [currentPage, allPlayers.length, fetchPlayers]);

  const filterAndPaginatePlayers = (players: Player[], page: number) => {
    const filteredPlayers = players.filter(
      (player) =>
        player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        player.role.toLowerCase().includes(searchTerm.toLowerCase())
    );
    const startIndex = (page - 1) * PLAYERS_PER_PAGE;
    return filteredPlayers.slice(startIndex, startIndex + PLAYERS_PER_PAGE);
  };

  const togglePlayerSelection = (player: Player, team: "A" | "B") => {
    if (team === "A") {
      if (teamAPlayers.some((p) => p.id === player.id)) {
        setTeamAPlayers((prev) => prev.filter((p) => p.id !== player.id));
      } else if (teamAPlayers.length < 11) {
        setTeamAPlayers((prev) => [...prev, player]);
        setTeamBPlayers((prev) => prev.filter((p) => p.id !== player.id));
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: "Maximum 11 players allowed in Team A",
        });
      }
    } else {
      if (teamBPlayers.some((p) => p.id === player.id)) {
        setTeamBPlayers((prev) => prev.filter((p) => p.id !== player.id));
      } else if (teamBPlayers.length < 11) {
        setTeamBPlayers((prev) => [...prev, player]);
        setTeamAPlayers((prev) => prev.filter((p) => p.id !== player.id));
      } else {
        toast({
          variant: "destructive",
          title: "Error",
          description: "Maximum 11 players allowed in Team B",
        });
      }
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = e.target?.result;
        const workbook = XLSX.read(data, { type: "binary" });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json<ExcelData>(worksheet);

        const requiredColumns = [
          "Player Name",
          "Squad",
          "Match Date",
          "Format",
        ];
        const missingColumns = requiredColumns.filter(
          (col) => !Object.keys(jsonData[0]).includes(col)
        );

        if (missingColumns.length > 0) {
          toast({
            variant: "destructive",
            title: "Error",
            description: `Missing compulsory columns: ${missingColumns.join(
              ", "
            )}. Please ensure all required columns are present.`,
          });
          return;
        }

        // if (jsonData.length > 0) {
        //   const firstRow = jsonData[0];
        //   const date = new Date(firstRow['Match Date']);
        //   // Set a random time between 9 AM and 5 PM
        //   date.setHours(9 + Math.floor(Math.random() * 8), Math.floor(Math.random() * 60), 0, 0);
        //   setSelectedDate(date);
        //   setMatchType(firstRow['Format'] as MatchType);
        // }

        const teamA = jsonData.slice(0, 11).map((row) => ({
          id: Math.random(), // Temporary ID
          name: row["Player Name"],
          role: row["Squad"],
        }));

        const teamB = jsonData.slice(11, 22).map((row) => ({
          id: Math.random(), // Temporary ID
          name: row["Player Name"],
          role: row["Squad"],
        }));

        setTeamAPlayers(teamA);
        setTeamBPlayers(teamB);
        setIsExcelMode(true);
        toast({
          title: "Success",
          description:
            "Excel file processed successfully. Teams have been populated.",
        });
      } catch (error) {
        console.error("Error processing Excel file:", error);
        toast({
          variant: "destructive",
          title: "Error",
          description: "Error processing Excel file. Please check the format.",
        });
      }
    };

    reader.readAsBinaryString(file);
  };

  const handleSubmit = async () => {
    if (teamAPlayers.length !== 11 || teamBPlayers.length !== 11) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Each team must have exactly 11 players",
      });
      return;
    }

    if (!selectedDate || !matchType) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Date and match type are required",
      });
      return;
    }

    if (!teamAName || !teamBName) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Team names are required",
      });
      return;
    }

    const submissionData = {
      date: selectedDate ? format(selectedDate, "yyyy-MM-dd") : "",
      match_type: matchType.toLowerCase(),
      team_a: teamAName,
      team_b: teamBName,
      team_a_players: teamAPlayers.map((p) => p.name),
      team_b_players: teamBPlayers.map((p) => p.name),
    };

    try {
      setIsLoading(true);
      const response = await axios.post(
        "https://qwertyweb.xyz:8443/backend/api/v1/add-match/",
        submissionData
      );
      toast({
        title: "Success",
        description: "Match created successfully!",
      });
      console.log(response.data);
    } catch (error) {
      console.error("Error creating match:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Error creating match. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const displayedPlayers = filterAndPaginatePlayers(allPlayers, currentPage);

  return (
    <div className="min-h-screen bg-gradient-to-b from-red-900 to-red-800 p-4">
      <div className="container mx-auto">
        <h1 className="text-2xl font-bold mb-4 text-white">Team Selection</h1>

        <div className="mb-6 flex items-center">
          <Input
            type="file"
            accept=".xlsx, .xls, .csv"
            onChange={handleFileUpload}
            className="bg-white text-gray-900"
          />
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" className="p-0 ml-2">
                  <HelpCircle className="h-4 w-4 text-white" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>
                  The Excel file must contain the following columns: Player
                  Name, Squad, Match Date, Format.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        <div className="flex space-x-4 mb-6">
          <Input
            type="text"
            placeholder="Team A Name"
            value={teamAName}
            onChange={(e) => setTeamAName(e.target.value)}
            className="bg-white text-gray-900"
          />
          <Input
            type="text"
            placeholder="Team B Name"
            value={teamBName}
            onChange={(e) => setTeamBName(e.target.value)}
            className="bg-white text-gray-900"
          />
        </div>

        <div className="flex space-x-4 mb-4">
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant={"outline"}
                className={`w-[240px] justify-start text-left font-normal bg-white text-gray-900`}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {selectedDate ? (
                  format(selectedDate, "yyyy-MM-dd")
                ) : (
                  <span>Pick a date</span>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0">
              <Calendar
                mode="single"
                selected={selectedDate}
                onSelect={(date) => setSelectedDate(date)}
                initialFocus
              />
            </PopoverContent>
          </Popover>

          <Select
            value={matchType}
            onValueChange={(value: MatchType) => setMatchType(value)}
          >
            <SelectTrigger className="w-[180px] bg-white text-gray-900">
              <SelectValue placeholder="Match Type" />
            </SelectTrigger>
            <SelectContent>
              {MATCH_TYPES.map((type) => (
                <SelectItem key={type} value={type}>
                  {type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {!isExcelMode && (
          <Input
            type="text"
            placeholder="Search players..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            className="mb-4 bg-white text-gray-900"
          />
        )}

        <div className="flex space-x-4">
          <div className="w-1/2 bg-white p-4 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Team A Players ({teamAPlayers.length}/11)
            </h2>
            <div className="h-[calc(100vh-350px)] overflow-y-auto">
              {(isExcelMode ? teamAPlayers : displayedPlayers).map(
                (player, index) => (
                  <Card key={`playerA-${player.id || index}`} className="mb-2">
                    <CardContent className="flex items-center justify-between p-4">
                      <span>
                        {player.name} ({player.role})
                      </span>
                      {!isExcelMode && (
                        <Checkbox
                          checked={teamAPlayers.some((p) => p.id === player.id)}
                          onCheckedChange={() =>
                            togglePlayerSelection(player, "A")
                          }
                        />
                      )}
                    </CardContent>
                  </Card>
                )
              )}
            </div>
          </div>

          <div className="w-1/2 bg-white p-4 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-2">
              Team B Players ({teamBPlayers.length}/11)
            </h2>
            <div className="h-[calc(100vh-350px)] overflow-y-auto">
              {(isExcelMode ? teamBPlayers : displayedPlayers).map(
                (player, index) => (
                  <Card key={`playerB-${player.id || index}`} className="mb-2">
                    <CardContent className="flex items-center justify-between p-4">
                      <span>
                        {player.name} ({player.role})
                      </span>
                      {!isExcelMode && (
                        <Checkbox
                          checked={teamBPlayers.some((p) => p.id === player.id)}
                          onCheckedChange={() =>
                            togglePlayerSelection(player, "B")
                          }
                        />
                      )}
                    </CardContent>
                  </Card>
                )
              )}
            </div>
          </div>
        </div>

        {!isExcelMode && (
          <div className="mt-4 flex justify-between items-center">
            <Button
              onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
            >
              Previous
            </Button>
            <span className="text-white">
              Page {currentPage} of {totalPages}
            </span>
            <Button
              onClick={() =>
                setCurrentPage((prev) => Math.min(prev + 1, totalPages))
              }
              disabled={currentPage === totalPages}
            >
              Next
            </Button>
          </div>
        )}

        <div className="mt-4 text-center">
          <Button
            onClick={handleSubmit}
            disabled={
              isLoading ||
              teamAPlayers.length !== 11 ||
              teamBPlayers.length !== 11 ||
              !teamAName ||
              !teamBName
            }
            className="bg-white text-gray-900 hover:bg-gray-100"
          >
            {isLoading ? <Loader className="animate-spin mr-2" /> : null}
            Create Match
          </Button>
        </div>
      </div>
    </div>
  );
}
