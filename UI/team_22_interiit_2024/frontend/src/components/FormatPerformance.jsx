import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

export default function FormatPerformanceChart({ data }) {

  return (
    <Card className="bg-white/20">
      <CardHeader>
        <CardTitle className="text-black">Format Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="text-black">Format</TableHead>
              <TableHead className="text-black">Lifetime Runs</TableHead>
              <TableHead className="text-black">Lifetime Wickets</TableHead>
              <TableHead className="text-black">Previous 3 Runs</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Object.entries(data)?.map(([format, stats]) => (
              <TableRow key={format}>
                <TableCell className="font-medium text-black">{format}</TableCell>
                <TableCell className="text-black">{stats["Lifetime Runs"]}</TableCell>
                <TableCell className="text-black">{stats["Lifetime Wickets"]}</TableCell>
                <TableCell className="text-black">{stats["Previous 3 Runs"].join(", ")}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

