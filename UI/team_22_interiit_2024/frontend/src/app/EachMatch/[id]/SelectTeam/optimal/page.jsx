
import { getMatchRelatedChats, getMatches } from "@/api";
import MatchDetailClient from "@/components/MatchDetailClient";
import Dream11AIChat from "@/components/chatbot";


export default async function Page({params}) {
  const { id } = params;
  const matchData = await getMatches({ id });
  const messages = await getMatchRelatedChats({match_id:id})

  return (
    <>
      <MatchDetailClient matchData={matchData[0]} />
      <Dream11AIChat messages = {messages} match_id={matchData[0].id} />
    </>
  );
}
