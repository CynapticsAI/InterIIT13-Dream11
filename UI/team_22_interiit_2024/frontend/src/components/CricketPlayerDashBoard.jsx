"use client";

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  ComposedChart,
  Line,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const colors = [
  "#271A45", // Deep Purple (Performance Metrics)
  "#4A3933", // Dark Brown (Played Against Runs)
  "#1F2937", // Dark Zinc (Club Performance)
  "#44403C",
  "#3F3F46", // Dark Gray/Zinc
  "#78350F", // Deep Brown (Alternate)
  "#1C1917", // Almost Black (Stone)
  "#292524", // Dark Stone Variant
  "#065F46", // Deep Teal (for contrast)
  "#14532D", // Dark Green (for contrast)
];

export default function ComprehensiveCricketPerformanceDashboard({
  playerStats,
  player_name,
}) {
  const [selectedPlayer, setSelectedPlayer] = useState(
    Object.keys(playerStats["Played Against"] || {})[0] || ""
  );

  // Transformed Data for All Sections (same as previous code)
  const performanceMetricsData = Object.entries(
    playerStats["Performance"] || {}
  ).map(([name, value]) => ({ name, value }));

  const playedAgainstData = Object.entries(
    playerStats["Played Against"] || {}
  ).map(([playerName, stats]) => ({
    name: playerName,
    ...stats,
  }));

  const selectedPlayerData = Object.entries(
    playerStats["Played Against"][selectedPlayer] || {}
  ).map(([name, value]) => ({ name, value }));

  const clubPerformanceData = Object.entries(
    playerStats["Club Performance"] || {}
  ).flatMap(([format, stats]) =>
    Object.entries(stats)
      .filter(([key]) => key !== "Previous 3 Runs")
      .map(([name, value]) => ({
        format,
        name,
        value: typeof value === "number" ? value : parseFloat(value),
      }))
  );

  const intlPerformanceData = Object.entries(
    playerStats["International Performance"] || {}
  ).flatMap(([format, stats]) =>
    Object.entries(stats)
      .filter(([key]) => key !== "Previous 3 Runs")
      .map(([name, value]) => ({
        format,
        name,
        value: typeof value === "number" ? value : parseFloat(value),
      }))
  );

  // Previous 3 Runs Data
  const clubPrevious3RunsData = Object.entries(
    playerStats["Club Performance"] || {}
  )
    .filter(([_, stats]) => stats["Previous 3 Runs"])
    .map(([format, stats]) => ({
      format,
      runs: stats["Previous 3 Runs"],
    }));

  const intlPrevious3RunsData = Object.entries(
    playerStats["International Performance"] || {}
  )
    .filter(([_, stats]) => stats["Previous 3 Runs"])
    .map(([format, stats]) => ({
      format,
      runs: stats["Previous 3 Runs"],
    }));

  return (
    <div className="min-h-screen bg-gradient-to-b from-red-900 to-red-800 p-8 w-screen">
      <Card className="w-full max-w-7xl mx-auto bg-white">
        <CardHeader className="bg-gray-100">
          <CardTitle>
            {player_name.replace("%20", " ")} Comprehensive Performance
            Dashboard
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="performanceMetrics" className="space-y-6">
            <TabsList className="grid grid-cols-5 bg-gray-200">
              <TabsTrigger value="performanceMetrics">
                Performance Metrics
              </TabsTrigger>
              <TabsTrigger value="playedAgainst">Played Against</TabsTrigger>
              <TabsTrigger value="clubPerformance">
                Club Performance
              </TabsTrigger>
              <TabsTrigger value="intlPerformance">
                International Performance
              </TabsTrigger>
              <TabsTrigger value="previous3Runs">Previous 3 Runs</TabsTrigger>
            </TabsList>

            {/* Performance Metrics Tab */}
            <TabsContent value="performanceMetrics" className="space-y-6">
              <div className="flex flex-col space-y-6">
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>Performance Metrics Bar Chart</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <BarChart
                      width={700}
                      height={300}
                      data={performanceMetricsData}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill={colors[0]} />
                    </BarChart>
                  </CardContent>
                </Card>
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>Performance Metrics Pie Chart</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <PieChart width={700} height={300}>
                      <Pie
                        data={performanceMetricsData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label
                      >
                        {performanceMetricsData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={colors[index % colors.length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Played Against Tab */}
            <TabsContent value="playedAgainst" className="space-y-6">
              <div className="mb-6">
                <Select
                  value={selectedPlayer}
                  onValueChange={setSelectedPlayer}
                >
                  <SelectTrigger className="w-[280px]">
                    <SelectValue placeholder="Select Opponent" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.keys(playerStats["Played Against"] || {}).map(
                      (player) => (
                        <SelectItem key={player} value={player}>
                          {player}
                        </SelectItem>
                      )
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex flex-col space-y-6">
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>Performance Against {selectedPlayer}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <BarChart
                      width={700}
                      height={300}
                      data={selectedPlayerData}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill={colors[1]} />
                    </BarChart>
                  </CardContent>
                </Card>
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>Overall Opponents Comparison</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ComposedChart
                      width={700}
                      height={300}
                      data={playedAgainstData}
                    >
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="Runs" barSize={20} fill={colors[4]} />
                      <Line
                        type="monotone"
                        dataKey="Wickets"
                        stroke={colors[5]}
                      />
                      <Legend />
                    </ComposedChart>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Club Performance Tab */}
            <TabsContent value="clubPerformance" className="space-y-6">
              <div className="flex flex-col space-y-6">
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>Club Performance Bar Chart</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <BarChart
                      width={700}
                      height={300}
                      data={clubPerformanceData}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill={colors[2]} />
                      <Legend />
                    </BarChart>
                  </CardContent>
                </Card>
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>Club Performance Radar Chart</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <RadarChart
                      width={700}
                      height={300}
                      data={clubPerformanceData}
                    >
                      <PolarGrid />
                      <PolarAngleAxis dataKey="name" />
                      <PolarRadiusAxis />
                      <Radar
                        dataKey="value"
                        stroke={colors[3]}
                        fill={colors[3]}
                        fillOpacity={0.6}
                      />
                      <Tooltip />
                    </RadarChart>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* International Performance Tab */}
            <TabsContent value="intlPerformance" className="space-y-6">
              <div className="flex flex-col space-y-6">
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>
                      International Performance Composed Chart
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ComposedChart
                      width={700}
                      height={300}
                      data={intlPerformanceData}
                    >
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="value" barSize={20} fill={colors[6]} />
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke={colors[7]}
                      />
                    </ComposedChart>
                  </CardContent>
                </Card>
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>International Performance Pie Chart</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <PieChart width={700} height={300}>
                      <Pie
                        data={intlPerformanceData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label
                      >
                        {intlPerformanceData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={colors[index % colors.length]}
                          />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Previous 3 Runs Tab */}
            <TabsContent value="previous3Runs" className="space-y-6">
              <div className="flex flex-col space-y-6">
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>Club Previous 3 Runs</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <BarChart
                      width={700}
                      height={300}
                      data={clubPrevious3RunsData}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="format" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="runs" fill={colors[8]} />
                    </BarChart>
                  </CardContent>
                </Card>
                <Card className="bg-white">
                  <CardHeader>
                    <CardTitle>International Previous 3 Runs</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <BarChart
                      width={700}
                      height={300}
                      data={intlPrevious3RunsData}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="format" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="runs" fill={colors[9]} />
                    </BarChart>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
