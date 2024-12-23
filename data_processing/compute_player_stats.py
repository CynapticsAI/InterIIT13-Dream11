from data_classes import Player


def calculate_player_data(stats, players_playing, match_type, match_tier, values, players):
    """
    Calculates the complete player wise data like his total runs in club/international matches , wickets ,
    head to head against all the matches he has played in his lifetime etc. and also appends values 
    to the row of the final data to be used as training data
    """
    for idx, row in stats.iterrows():
        player_id = players_playing[idx]
        if player_id not in players.keys():
            players[player_id] = Player()

        player: Player = players[player_id]

        for tier in player.match_tiers:
            for types in player.match_types:
                for duration_type in player.duration:
                    for attr in player.player_attributes:
                        if attr == 'strike rate':
                            if duration_type == 'lifetime':
                                runs = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_runs')
                                balls = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_balls played')
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                            else:
                                runs_list = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_runs')
                                balls_list = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_balls played')
                                runs = sum(runs_list)
                                balls = sum(balls_list)
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                        elif attr == 'economy':
                            if duration_type == 'lifetime':
                                runs_gave = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_runs gave')
                                overs = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_overs')
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                            else:
                                runs_gave_list = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_runs gave')
                                overs_list = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_overs')
                                runs_gave = sum(runs_gave_list)
                                overs = sum(overs_list)
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                        else:
                            if duration_type == 'lifetime':
                                values.append(player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_' + attr))
                            else:
                                curr = player.get_variable(
                                    tier + '_' + types + '_' + duration_type + '_' + attr)
                                tot = sum(curr)
                                values.append(tot)

        s = match_tier + '_' + match_type + '_'

        curr_runs = row['runs']
        player.runs += curr_runs
        player.wickets += row['wickets']

        if curr_runs >= 100:
            player.centuries += 1
        elif curr_runs >= 50:
            player.fifties += 1
        elif curr_runs >= 30:
            player.thirties += 1

        player.maidens += row['maidens']
        if row['wickets'] >= 3:
            player.hattricks += 1

        for attr in player.player_attr:
            curr = player.get_variable(s + 'lifetime_' + attr)
            new_val = row[attr]
            player.set_variable(s + 'lifetime_' + attr, curr + new_val)

        for attr in player.player_attr:
            curr: list = player.get_variable(s + 'previous3_' + attr)
            if len(curr) == 3:
                curr.pop(0)

            curr.append(row[attr])
            player.set_variable(s + 'previous3_' + attr, curr)

        players[player_id] = player
