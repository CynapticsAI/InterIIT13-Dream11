import React from 'react'
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function PerformanceChart({ data }) {
  const chartData = [
    { name: 'Runs', value: data?.Runs },
    { name: 'Balls', value: data?.Balls },
    { name: 'Wickets', value: data?.Wickets * 10 } // Multiplying by 10 to make it visible on the chart
  ]

  return (
    <Card className="bg-white/20 text-black">
      <CardHeader>
        <CardTitle className="text-black">Performance Chart</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300} className={"!text-black"}>
          <BarChart data={chartData} className='invert'>
            <XAxis dataKey="name" stroke="#ffffff" />
            <YAxis stroke="#ffffff" />
            <Bar dataKey="value" fill="#ef4444" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

