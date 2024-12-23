import json
import os
import numpy as np
from data_classes import Player, Player_against
from compute_match_stats import calc_stats
from compute_player_stats import calculate_player_data


def dict_to_Played_Against(data):
    """Converts the dictionary object of the played against form to an object of Player Against class"""
    played_against = Player_against()
    for attr in data.keys():
        played_against.set_variable(attr, data[attr])

    return played_against


def dict_to_Player(data):
    """Converts the dictionary object stored in data/processed/players files to an object of Player class"""
    player = Player()
    for attr, val in data.items():
        if attr == 'played_against':
            for other_player, other_player_data in val.items():
                player.played_against[other_player] = dict_to_Played_Against(
                    other_player_data)
        else:
            player.set_variable(attr, data[attr])

    return player


def add_new_data(file):
    """This function updates the data of the players stored in data/processes/players according to the new match file given"""
    with open(file, 'r') as f:
        data = json.load(f)

    team1 = []
    team2 = []
    values = []
    match_type = data['info']['match_type']
    match_tier = data['info']['team_type']
    players_playing = data['info']['registry']['people']
    i = 0
    for team in data['info']['players'].values():
        if i == 0:
            for idx in range(0, 11):
                team1.append(team[idx])
                player_id = players_playing[team[idx]]
                try:
                    with open(player_data_dst + player_id + '.json') as f:
                        file_data = json.load(f)
                    players[player_id] = dict_to_Player(file_data)
                except:
                    pass
        else:
            for idx in range(0, 11):
                team2.append(team[idx])
                player_id = players_playing[team[idx]]
                try:
                    with open(player_data_dst + player_id + '.json') as f:
                        file_data = json.load(f)
                    players[player_id] = dict_to_Player(file_data)
                except:
                    pass
        i += 1

    num_balls_per_over = data['info']['balls_per_over']

    stats = calc_stats(data, values, team1, team2,
                       num_balls_per_over, players_playing, players)
    calculate_player_data(stats, players_playing,
                          match_type, match_tier, values, players)

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
        with open(player_data_dst + 'playerssss' + player + '.json', 'w') as f:
            f.write(json.dumps(player_data))


def main():
    global players, player_data_dst
    players = {}
    current_path = os.getcwd()
    parent_path = os.path.dirname(current_path)
    player_data_dst = os.path.join(parent_path, 'data/processed/players/')
    add_new_data('../data/raw/64815.json')


main()
