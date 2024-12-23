
import { getMatches } from "@/api";
import EachMatchCard from "@/components/EachMatchCard";


export default async function Page({params}) {
  const { id } = params;
  const match = await getMatches({ id:id });

  return <EachMatchCard match={match} id={id} />;
}
