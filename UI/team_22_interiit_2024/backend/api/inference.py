from backend.settings import BASE_DIR
import json
import os
import pandas as pd
match_tiers = ['club' , 'international']
match_types = ['ODI' , 'Test' , 'T20']
duration = ['previous3' , 'lifetime']
h2h_attr = ['runs' , 'balls' , 'wickets']
player_attributes = ['runs' , 'wickets' , 'balls played' , 'economy' , 'strike rate' ]
fantasy_points_attr = ['runs' , 'wickets' , '4s' , '6s' , 'catches' , 'maidens'  , 'overs' ,
                          'runs gave' , 'out' , 'ducks' ,'balls played' , 'lbw/bowled' ,'runout(indirect)' , 'runout(direct)', 'fantasy points' ]
cols = ['date' , 'ODI' , 'Test' , 'T20' , 'venue' , 'club' , 'international' , 'match_state' ]
match_tier = 'international'


def get_response(team1_names,team2_names,match_type,date):
    values = []
    values.append(date)
    for type in match_types :
        if match_type == type :
            values.append(1)
        else :
            values.append(0)

    #venue
    values.append(0)

    #match tier
    for tier in match_tiers :
        if tier == match_tier :
            values.append(1)
        else :
            values.append(0)

    #match state
    values.append(1)

    player_data = {}
    with open(os.path.join(BASE_DIR,'data/player_id.json'), 'r') as f :
        player_ids = json.load(f)

    team1 = []
    team2 = []

    #team players
    for idx in range(0 , 11) :
        values.append(team1_names[idx])
        player_id = player_ids[team1_names[idx]]
        team1.append(player_id)
        with open(os.path.join(BASE_DIR,'data/players/players/' + player_id + '.json')) as f :
            player_data[player_id] = json.load(f)
    for idx in range(0 , 11) :
        values.append(team2_names[idx])
        player_id = player_ids[team2_names[idx]]
        team2.append(player_id)
        with open(os.path.join(BASE_DIR,'data/players/players/' + player_id + '.json')) as f :
            player_data[player_id] = json.load(f)

    #current match stats(fantasy points related)
    for i in range(0 ,22) :
        for attr in fantasy_points_attr :
            values.append(0)

    #head to head
    for i in range(0 , 11) :
        team1_player_obj = player_data[team1[i]]
        for j in range(0 , 11) :
            team2_player_id = team2[j]
            if team2_player_id in team1_player_obj['played_against'].keys() :
                team2_player_obj = team1_player_obj['played_against'][team2_player_id]
                values.append(team2_player_obj['runs'])
                values.append(team2_player_obj['balls'])
                values.append(team2_player_obj['wickets'])
            else :
                for k in range(0, 3) :
                    values.append(0)

    for i in range(0 , 11) :
        team2_player_obj = player_data[team2[i]]
        for j in range(0 , 11) :
            team1_player_id = team1[j]
            if team1_player_id in team2_player_obj['played_against'].keys() :
                team1_player_obj = team2_player_obj['played_against'][team1_player_id]
                values.append(team1_player_obj['runs'])
                values.append(team1_player_obj['balls'])
                values.append(team1_player_obj['wickets'])
            else :
                for k in range(0, 3) :
                    values.append(0)

    for player_id in team1 :
        player = player_data[player_id]
        for tier in match_tiers :
            for type in match_types :
                for duration_type in duration :
                    for attr in player_attributes :
                        if attr == 'strike rate':
                            if duration_type == 'lifetime' :
                                runs = player[tier + '_' + type + '_' + duration_type + '_runs']
                                balls = player[tier + '_' + type + '_' + duration_type + '_balls played']
                                if balls == 0:
                                    values.append(0)
                                else :
                                    values.append(runs*100.0/balls)
                            else :
                                runs_list = player[tier + '_' + type + '_' + duration_type + '_runs']
                                balls_list = player[tier + '_' + type + '_' + duration_type + '_balls played']
                                runs = sum(runs_list)
                                balls = sum(balls_list)
                                if balls == 0:
                                    values.append(0)
                                else :
                                    values.append(runs*100.0/balls)
                        elif attr == 'economy' :
                            if duration_type == 'lifetime' :
                                runs_gave = player[tier + '_' + type + '_' + duration_type + '_runs gave']
                                overs = player[tier + '_' + type + '_' + duration_type + '_overs']
                                if overs == 0:
                                    values.append(100)
                                else :
                                    values.append(runs_gave*1.0/overs)
                            else :
                                runs_gave_list = player[tier + '_' + type + '_' + duration_type + '_runs gave']
                                overs_list = player[tier + '_' + type + '_' + duration_type + '_overs']
                                runs_gave = sum(runs_gave_list)
                                overs = sum(overs_list)
                                if overs == 0:
                                    values.append(100)
                                else :
                                    values.append(runs_gave*1.0/overs)
                        else :
                            if duration_type == 'lifetime' :
                                values.append(player[tier + '_' + type + '_' + duration_type + '_' + attr])
                            else :
                                curr = player[tier + '_' + type + '_' + duration_type + '_' + attr]
                                tot = sum(curr)
                                values.append(tot)

    for player_id in team2 :
        player = player_data[player_id]
        for tier in match_tiers :
            for type in match_types :
                for duration_type in duration :
                    for attr in player_attributes :
                        if attr == 'strike rate':
                            if duration_type == 'lifetime' :
                                runs = player[tier + '_' + type + '_' + duration_type + '_runs']
                                balls = player[tier + '_' + type + '_' + duration_type + '_balls played']
                                if balls == 0:
                                    values.append(0)
                                else :
                                    values.append(runs*100.0/balls)
                            else :
                                runs_list = player[tier + '_' + type + '_' + duration_type + '_runs']
                                balls_list = player[tier + '_' + type + '_' + duration_type + '_balls played']
                                runs = sum(runs_list)
                                balls = sum(balls_list)
                                if balls == 0:
                                    values.append(0)
                                else :
                                    values.append(runs*100.0/balls)
                        elif attr == 'economy' :
                            if duration_type == 'lifetime' :
                                runs_gave = player[tier + '_' + type + '_' + duration_type + '_runs gave']
                                overs = player[tier + '_' + type + '_' + duration_type + '_overs']
                                if overs == 0:
                                    values.append(100)
                                else :
                                    values.append(runs_gave*1.0/overs)
                            else :
                                runs_gave_list = player[tier + '_' + type + '_' + duration_type + '_runs gave']
                                overs_list = player[tier + '_' + type + '_' + duration_type + '_overs']
                                runs_gave = sum(runs_gave_list)
                                overs = sum(overs_list)
                                if overs == 0:
                                    values.append(100)
                                else :
                                    values.append(runs_gave*1.0/overs)
                        else :
                            if duration_type == 'lifetime' :
                                values.append(player[tier + '_' + type + '_' + duration_type + '_' + attr])
                            else :
                                curr = player[tier + '_' + type + '_' + duration_type + '_' + attr]
                                tot = sum(curr)
                                values.append(tot)
    print('values',values)
    return values