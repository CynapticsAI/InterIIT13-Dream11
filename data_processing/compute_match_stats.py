import pandas as pd
import copy
from calculate_fantasy_points import calc_fantasy_points, rules
from data_classes import Player, Player_against


def calc_stats(data, values, team1, team2, num_balls_per_over, players_playing, players):
    """
    The function takes the json of a match as input and computes the stats for the match for each individual
    and the fantasy points as well using the calc_fantasy_points function 
    This function also calculates the head to head to head stats of each player of team1 wrt to players of team2
    All the stats are pushed into the values list which forms the row of the final csv to train the model  
    """

    df = pd.DataFrame(columns=['name', 'runs', 'wickets', '4s', '6s', 'catches', 'maidens', 'overs',
                               'runs gave', 'out', 'ducks',  'balls played', 'lbw/bowled', 'runout(indirect)', 'runout(direct)', 'fantasy points'])

    df['overs'] = df['overs'].astype(float)

    for player in team1:
        df.loc[len(df)] = [player, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        player_id = players_playing[player]
        if player_id not in players.keys():
            players[player_id] = Player()
    for player in team2:
        df.loc[len(df)] = [player, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        player_id = players_playing[player]
        if player_id not in players.keys():
            players[player_id] = Player()

    for i in range(330):
        values.append(0)

    for i in range(0, 11):
        team1_player_obj: Player = players[players_playing[team1[i]]]
        for j in range(0, 11):
            team2_player_id = players_playing[team2[j]]
            if team2_player_id in team1_player_obj.played_against.keys():
                team2_player_obj: Player_against = team1_player_obj.played_against[
                    team2_player_id]
                values.append(team2_player_obj.runs)
                values.append(team2_player_obj.balls)
                values.append(team2_player_obj.wickets)
            else:
                for k in range(0, 3):
                    values.append(0)
    for i in range(0, 11):
        for j in range(0, 11):
            team2_player_obj: Player = players[players_playing[team2[i]]]
            team1_player_id = players_playing[team1[j]]
            if team1_player_id in team2_player_obj.played_against.keys():
                team1_player_obj: Player_against = team2_player_obj.played_against[
                    team1_player_id]
                values.append(team1_player_obj.runs)
                values.append(team1_player_obj.balls)
                values.append(team1_player_obj.wickets)
            else:
                for k in range(0, 3):
                    values.append(0)

    df.set_index('name', inplace=True)
    df_final = copy.deepcopy(df)

    for inning in data['innings']:
        if 'overs' not in inning.keys():
            continue
        for over in inning['overs']:
            runs = 0
            fair_deliveries = 0
            for delivery in over['deliveries']:
                batter = delivery['batter']
                bowler = delivery['bowler']

                batter_obj: Player = players[players_playing[batter]]
                bowler_id = players_playing[bowler]
                if bowler_id not in batter_obj.played_against.keys():
                    batter_obj.played_against[bowler_id] = Player_against()
                bowler_obj: Player_against = batter_obj.played_against[bowler_id]

                r = delivery['runs']['batter']
                df.loc[batter]['runs'] += r
                bowler_obj.runs += r
                if r == 4:
                    df.loc[batter]['4s'] += 1
                elif r == 6:
                    df.loc[batter]['6s'] += 1

                runs += delivery['runs']['total']

                if 'extras' in delivery.keys():
                    extra_types = delivery['extras'].keys()
                    if 'legbyes' in extra_types and len(extra_types) == 1:
                        df.loc[batter]['balls played'] += 1
                        fair_deliveries += 1
                else:
                    df.loc[batter]['balls played'] += 1
                    fair_deliveries += 1
                    bowler_obj.balls += 1

                if 'wickets' in delivery.keys():
                    for wicket in delivery['wickets']:
                        player = wicket['player_out']
                        df.loc[player]['out'] += 1
                        if df.loc[player]['runs'] == 0:
                            df.loc[player]['ducks'] += 1
                        if wicket['kind'] == 'caught':
                            df.loc[bowler]['wickets'] += 1
                            bowler_obj.wickets += 1
                            if 'name' not in wicket['fielders'][0].keys() or 'substitute' in wicket['fielders'][0].keys():
                                continue
                            fielder = wicket['fielders'][0]['name']
                            df.loc[fielder]['catches'] += 1
                        elif wicket['kind'] == 'caught and bowled':
                            df.loc[bowler]['wickets'] += 1
                            df.loc[bowler]['catches'] += 1
                            bowler_obj.wickets += 1
                        elif wicket['kind'] == 'bowled' or wicket['kind'] == 'bowled':
                            df.loc[bowler]['wickets'] += 1
                            df.loc[bowler]['lbw/bowled'] += 1
                            bowler_obj.wickets += 1
                        elif wicket['kind'] == 'run out':
                            if 'fielders' in wicket.keys():
                                if len(wicket['fielders']) == 2:
                                    fielder1 = wicket['fielders'][0]['name']
                                    fielder2 = wicket['fielders'][1]['name']
                                    if 'substitute' in wicket['fielders'][0].keys():
                                        pass
                                    else:
                                        df.loc[fielder1]['runout(indirect)'] += 1
                                    if 'substitute' in wicket['fielders'][1].keys():
                                        pass
                                    else:
                                        df.loc[fielder2]['runout(indirect)'] += 1
                                else:
                                    if 'name' not in wicket['fielders'][0].keys() or 'substitute' in wicket['fielders'][0].keys():
                                        continue
                                    fielder = wicket['fielders'][0]['name']
                                    df.loc[fielder]['runout(direct)'] += 1
                            else:
                                df.loc[bowler]['runout(direct)'] += 1

            overs = fair_deliveries*1.0/num_balls_per_over

            if overs == 1 and runs == 0:
                df.loc[bowler]['maidens'] += 1

            df.loc[bowler]['overs'] += float(overs)
            df.loc[bowler]['runs gave'] += runs

        calc_fantasy_points(df, rules[data['info']['match_type']])
        df_final = df_final + df
        columns_to_zero = df.columns[:]
        df[columns_to_zero] = 0.0

    j = 0
    for idx, row in df_final.iterrows():
        if idx in team1 and team1.index(idx) < 11:
            for col_name, value in row.items():
                values[j + 30] = value
                j += 1
        elif idx in team2 and team2.index(idx) < 11:
            for col_name, value in row.items():
                values[j + 30] = value
                j += 1
        else:
            df_final.drop(index=idx, inplace=True)

    return df_final
