import pandas as pd
import numpy as np
import json
import pickle

match_tiers = ['club', 'international']
match_types = ['ODI', 'Test', 'T20']
duration = ['previous3', 'lifetime']
h2h_attr = ['runs', 'balls', 'wickets']
player_attributes = ['runs', 'wickets',
                     'balls played', 'economy', 'strike rate']
fantasy_points_attr = ['runs', 'wickets', '4s', '6s', 'catches', 'maidens', 'overs',
                       'runs gave', 'out', 'ducks', 'balls played', 'lbw/bowled', 'runout(indirect)', 'runout(direct)', 'fantasy points']
cols = ['date', 'ODI', 'Test', 'T20', 'venue',
        'club', 'international', 'match_state']
team1_names = ['NG Jones', 'C Dougherty', 'LT Nelson', 'Mansoor Amjad', 'J Holmes',
               'NL Smith', 'GJ Thompson', 'PA Eakin', 'GJ McCarter', 'GE Kidd', 'PS Eaglestone']
team2_names = ['DI Joyce', 'FP McAllister', 'J Anderson', 'Simi Singh', 'AD Poynter',
               "KJ O'Brien", 'TE Kane', 'EJ Richardson', 'GH Dockrell', 'MC Sorensen', 'Yaqoob Ali']

date = '2014'
match_tier = 'international'


def predict(team1_names, team2_names, match_type):
    """
    The function needs an input in the form of list of names for team1 and team2 players
    and it outputs the best playing 11 according to the model stored in model.pkl
    """
    values = []
    for type in match_types:
        if match_type == type:
            values.append(1)
        else:
            values.append(0)

    # venue
    values.append(0)

    # match tier
    for tier in match_tiers:
        if tier == match_tier:
            values.append(1)
        else:
            values.append(0)

    # match state
    values.append(1)

    player_data = {}
    with open('../data/processed/player_id.json', 'r') as f:
        player_ids = json.load(f)

    team1 = []
    team2 = []

    all_names = team1_names + team2_names

    # team players
    for idx in range(0, 11):
        player_id = player_ids[team1_names[idx]]
        team1.append(player_id)
        with open('../data/processed/players/' + player_id + '.json') as f:
            player_data[player_id] = json.load(f)
    for idx in range(0, 11):
        player_id = player_ids[team2_names[idx]]
        team2.append(player_id)
        with open('../data/processed/players/' + player_id + '.json') as f:
            player_data[player_id] = json.load(f)


    # head to head
    for i in range(0, 11):
        team1_player_obj = player_data[team1[i]]
        for j in range(0, 11):
            team2_player_id = team2[j]
            if team2_player_id in team1_player_obj['played_against'].keys():
                team2_player_obj = team1_player_obj['played_against'][team2_player_id]
                values.append(team2_player_obj['runs'])
                values.append(team2_player_obj['balls'])
                values.append(team2_player_obj['wickets'])
            else:
                for k in range(0, 3):
                    values.append(0)

    for i in range(0, 11):
        team2_player_obj = player_data[team2[i]]
        for j in range(0, 11):
            team1_player_id = team1[j]
            if team1_player_id in team2_player_obj['played_against'].keys():
                team1_player_obj = team2_player_obj['played_against'][team1_player_id]
                values.append(team1_player_obj['runs'])
                values.append(team1_player_obj['balls'])
                values.append(team1_player_obj['wickets'])
            else:
                for k in range(0, 3):
                    values.append(0)

    for player_id in team1:
        player = player_data[player_id]
        for tier in match_tiers:
            for type in match_types:
                for duration_type in duration:
                    for attr in player_attributes:
                        if attr == 'strike rate':
                            if duration_type == 'lifetime':
                                runs = player[tier + '_' + type +
                                              '_' + duration_type + '_runs']
                                balls = player[tier + '_' + type +
                                               '_' + duration_type + '_balls played']
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                            else:
                                runs_list = player[tier + '_' +
                                                   type + '_' + duration_type + '_runs']
                                balls_list = player[tier + '_' + type +
                                                    '_' + duration_type + '_balls played']
                                runs = sum(runs_list)
                                balls = sum(balls_list)
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                        elif attr == 'economy':
                            if duration_type == 'lifetime':
                                runs_gave = player[tier + '_' + type +
                                                   '_' + duration_type + '_runs gave']
                                overs = player[tier + '_' + type +
                                               '_' + duration_type + '_overs']
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                            else:
                                runs_gave_list = player[tier + '_' +
                                                        type + '_' + duration_type + '_runs gave']
                                overs_list = player[tier + '_' +
                                                    type + '_' + duration_type + '_overs']
                                runs_gave = sum(runs_gave_list)
                                overs = sum(overs_list)
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                        else:
                            if duration_type == 'lifetime':
                                values.append(
                                    player[tier + '_' + type + '_' + duration_type + '_' + attr])
                            else:
                                curr = player[tier + '_' + type +
                                              '_' + duration_type + '_' + attr]
                                tot = sum(curr)
                                values.append(tot)

    for player_id in team2:
        player = player_data[player_id]
        for tier in match_tiers:
            for type in match_types:
                for duration_type in duration:
                    for attr in player_attributes:
                        if attr == 'strike rate':
                            if duration_type == 'lifetime':
                                runs = player[tier + '_' + type +
                                              '_' + duration_type + '_runs']
                                balls = player[tier + '_' + type +
                                               '_' + duration_type + '_balls played']
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                            else:
                                runs_list = player[tier + '_' +
                                                   type + '_' + duration_type + '_runs']
                                balls_list = player[tier + '_' + type +
                                                    '_' + duration_type + '_balls played']
                                runs = sum(runs_list)
                                balls = sum(balls_list)
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                        elif attr == 'economy':
                            if duration_type == 'lifetime':
                                runs_gave = player[tier + '_' + type +
                                                   '_' + duration_type + '_runs gave']
                                overs = player[tier + '_' + type +
                                               '_' + duration_type + '_overs']
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                            else:
                                runs_gave_list = player[tier + '_' +
                                                        type + '_' + duration_type + '_runs gave']
                                overs_list = player[tier + '_' +
                                                    type + '_' + duration_type + '_overs']
                                runs_gave = sum(runs_gave_list)
                                overs = sum(overs_list)
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                        else:
                            if duration_type == 'lifetime':
                                values.append(
                                    player[tier + '_' + type + '_' + duration_type + '_' + attr])
                            else:
                                curr = player[tier + '_' + type +
                                              '_' + duration_type + '_' + attr]
                                tot = sum(curr)
                                values.append(tot)
    
    with open('../model_artifacts/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    y_pred = loaded_model.predict([values])

    sortex_idx = np.argsort(y_pred[0])[::-1]

    prediction = []
    for i in range(11):
        prediction.append(all_names[sortex_idx[i]])

    print(prediction)

    return prediction

predict(team1_names , team2_names , 'ODI')