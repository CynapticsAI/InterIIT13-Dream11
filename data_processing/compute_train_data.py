import pandas as pd
import numpy as np
import json
import os
import copy
from compute_match_stats import calc_stats
from compute_player_stats import calculate_player_data

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
raw_data_path = os.path.join(parent_path, 'data/raw/')
venue_data_path = os.path.join(parent_path, 'data/interim/venue_code.csv')
dst_file = os.path.join(parent_path, 'data/processed/match_data.csv')
player_data_dst = os.path.join(parent_path, 'data/processed/players/')

# Sorting
matches = {}
files = os.listdir(raw_data_path)

files = files[0:100]
for file in files:
    file_name = raw_data_path + file
    with open(file_name, "r") as f:
        data = json.load(f)
    matches[file] = data['info']['dates'][0]

files = []
for key in sorted(matches, key=matches.get):
    files.append(key)

venue_data = pd.read_csv(venue_data_path, index_col='Venue')

players = {}
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

for j in range(1, 3):
    for i in range(1, 12):
        cols.append('team' + str(j) + '_' + 'player' + str(i))
for j in range(1, 3):
    for i in range(1, 12):
        for attr in fantasy_points_attr:
            cols.append('team' + str(j) + '_' + 'player' + str(i) + '_' + attr)

for i in range(0, 11):
    for j in range(0, 11):
        for attr in h2h_attr:
            cols.append('team1_player' + str(i+1) +
                        '_team2_player' + str(j+1) + '_' + attr)

for i in range(0, 11):
    for j in range(0, 11):
        for attr in h2h_attr:
            cols.append('team2_player' + str(i+1) +
                        '_team1_player' + str(j+1) + '_' + attr)

for j in range(1, 3):
    for i in range(1, 12):
        for tier in match_tiers:
            for types in match_types:
                for duration_type in duration:
                    for attr in player_attributes:
                        cols.append('team' + str(j) + '_' + 'player' + str(i) + '_' +
                                    tier + '_' + types + '_' + duration_type + '_' + attr)

df = pd.DataFrame(columns=cols)


def compute_data():
    """
    Computes the final train data dataframe to be used as the training data 
    using the calc_stats and calc_player_stats 
    Also compute the json files for each player which store the complete data of the player like his
    lifetime runs , wickets , balls played and the stats against all players he had played
    """
    files_done = 0
    for file in files:
        print(files_done, file)
        file_name = raw_data_path + file
        with open(file_name, "r") as f:
            data = json.load(f)

        num_balls_per_over = data['info']['balls_per_over']
        values = []

        # date
        values.append(data['info']['dates'][0])

        # match types
        match_type = data['info']['match_type']
        if match_type == 'MDM':
            match_type = 'Test'
        if match_type == 'ODM':
            match_type = 'ODI'
        if match_type == 'IT20':
            match_type = 'T20'
        for types in match_types:
            if match_type == types:
                values.append(1)
            else:
                values.append(0)

        # venue
        location = data['info']['venue']
        val = venue_data.loc[location]['Cluster_Assignment']
        values.append(val)

        # club/international
        match_tier = data['info']['team_type']
        for tier in match_tiers:
            if tier == match_tier:
                values.append(1)
            else:
                values.append(0)

        # match state
        if len(data['innings']) == 0:
            pass
        else:
            if data['info']['teams'][0] == data['innings'][0]['team']:
                values.append(1)
            else:
                values.append(-1)

        # team players
        players_playing = data['info']['registry']['people']
        i = 0
        for team in data['info']['players'].values():
            if i == 0:
                team1 = team
            else:
                team2 = team
            i += 1
        for idx in range(0, 11):
            values.append(team1[idx])
        for idx in range(0, 11):
            values.append(team2[idx])

        # current match stats(fantasy points related)
        stats = calc_stats(data, values, team1, team2,
                           num_balls_per_over, players_playing, players)

        # all stats of player
        calculate_player_data(stats, players_playing,
                              match_type, match_tier, values, players)

        df.loc[len(df)] = copy.deepcopy(values)

        files_done += 1

    df.to_csv(dst_file, index=False)

    player_info_data = {}
    for player in players.keys():
        player_info = players[player].__dict__
        for other_player in player_info["played_against"]:
            try:
                player_info["played_against"][other_player] = player_info["played_against"][other_player].__dict__
            except:
                continue
        for key in player_info:
            if type(player_info[key]) not in [int, dict, str]:
                if type(player_info[key]) == list:
                    temp = []
                    for v in player_info[key]:
                        if type(v) == np.int64:
                            temp.append(int(v))
                        elif type(v) == np.float64:
                            temp.append(float(v))
                    player_info[key] = temp
                else:
                    if type(player_info[key]) == np.int64:
                        player_info[key] = int(player_info[key])
                    elif type(player_info[key]) == np.float64:
                        player_info[key] = float(player_info[key])
        player_info_data[player] = player_info

    for player in player_info_data.keys():
        player_data = player_info_data[player]
        with open(player_data_dst + player + '.json', 'w') as f:
            f.write(json.dumps(player_data))


compute_data()
