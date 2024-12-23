import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import Link from "next/link";
import { MatchFC } from "@/types";

interface MatchCardProps {
  match: MatchFC;
}

export default function MatchCard({ match }: MatchCardProps) {
  return (
    <Card className="w-full min-w-[300px] max-w-md flex-none">
      <CardContent className="p-4">
        <h3 className="mb-4 text-center text-lg font-semibold text-gray-600">{`${match.team_a.name} vs ${match.team_b.name} T20I`}</h3>
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-12 w-12 rounded-full bg-gray-100" />
            <span className="text-xl font-bold">{match.team_a.name}</span>
          </div>
          <div className="rounded-full bg-red-50 px-4 py-2 text-red-600">
            <span className="font-semibold">{match.date}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold">{match.team_b.name}</span>
            <div className="h-12 w-12 rounded-full bg-gray-100" />
          </div>
        </div>
        <div className="mb-4 grid grid-cols-2 gap-4 text-center">
          <div>
            <p className="text-sm text-gray-600">Prize Pool</p>
            <p className="text-xl font-bold">1,00,00,000</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">1st Prize</p>
            <p className="text-xl font-bold">50,00,000</p>
          </div>
        </div>
        <div className="mb-4 flex justify-between text-sm text-gray-600">
          <span>27 spots left</span>
          <span>58 spots</span>
        </div>
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600">Join for</p>
            <p className="text-lg">
              <span className="text-gray-400 line-through">₹49</span>{" "}
              <span className="font-bold">₹1 Only</span>
            </p>
          </div>
          <Link href={`/EachMatch/${match.id}`} className="w-fit h-fit">
            <Button className="bg-green-600 px-8 hover:bg-green-700">
              JOIN NOW
            </Button>
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
