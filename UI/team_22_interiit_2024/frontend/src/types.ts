export interface Player {
  name: string;
  unique_name: string;
  identifier: string;
  role: string;
}

export interface Team {
  id: number;
  players: Player[];
  name: string;
}

export interface Outcome {
  runs: number;
  wickets: number | null;
  winner: string;
  match_info: number;
}

export interface Extra {
  type: string;
  runs: number;
}

export interface Wicket {
  player_out: string;
  kind: string;
  fielders: string[];
}

export interface Delivery {
  ball_number: number;
  runs: number;
  extras: Extra;
  wickets: Wicket[];
}

export interface Over {
  over_number: number;
  deliveries: Delivery[];
}

export interface Inning {
  id: number;
  team: Team;
  outcome: Outcome;
  overs: Over[];
}

export interface Powerplay {
  type: string;
  overs: number;
}

export interface MatchFC {
  id: number;
  team_a: Team;
  team_b: Team;
  outcome: Outcome;
  balls_per_over: number;
  city: string;
  date: string;
  gender: string;
  match_type: string;
  match_type_number: number;
  overs: number;
  season: string;
  team_type: string;
  venue: string;
  player_of_match: string;
  toss_decision: string;
  toss_winner: string;
  target_runs: number;
  target_overs: number;
  meta: number | object;
  innings?: Inning[];
  powerplays?: Powerplay[];
  team_a_players?: Player[] | undefined;
  team_b_players?: Player[] | undefined;
}
