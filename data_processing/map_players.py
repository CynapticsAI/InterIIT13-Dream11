"""Create a mapping of the player names with their ids"""
import pandas as pd
import json
import os

files = os.listdir("../data/raw")

players = {}

for file in files:
    file_name = "../data/raw/" + file
    with open(file_name, "r") as f:
        data = json.load(f)

    players_playing = data['info']['registry']['people']
    for team in data['info']['players'].values():
        for player in team:
            player_id = players_playing[player]
            if player not in players:
                players[player] = player_id

with open("../data/processed/player_id.json", "w") as f:
    f.write(json.dumps(players))
