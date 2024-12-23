import { getMatchRelatedChats, getMatches } from "@/api";
import MatchDetailClient from "@/components/MatchDetailClient";
import Dream11AIChat from "@/components/chatbot";

export default async function Page({ params }) {
  const { id } = params;
  const matchData = await getMatches({ id });
  const messages = await getMatchRelatedChats({ match_id: id });

  // Filter team_a_players and team_b_players
  const team_a_player_ids = matchData[0].team_a_players?.map(player => player.identifier) || [];
  const team_b_player_ids = matchData[0].team_b_players?.map(player => player.identifier) || [];
  console.log(team_a_player_ids)
  console.log(team_b_player_ids)
  return (
    <>
      <MatchDetailClient matchData={matchData[0]} />
      <Dream11AIChat 
        messages={messages} 
        match_id={matchData[0].id} 
        team_a_player={team_a_player_ids}
        team_b_player={team_b_player_ids}
      />
    </>
  );
}
