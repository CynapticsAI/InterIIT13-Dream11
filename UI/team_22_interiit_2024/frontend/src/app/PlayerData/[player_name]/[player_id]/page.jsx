
import { getPlayerData } from "@/api.js";

import CricketPlayerDashboard from "@/components/CricketPlayerDashBoard";

export default async function Page({ params }) {
  const { player_id, player_name } = params;
  const playerStats = await getPlayerData({ player_identifier: player_id });


  return (
    <div className="container">
      <CricketPlayerDashboard playerStats={playerStats} player_name={player_name} />
    </div>
  );
}
