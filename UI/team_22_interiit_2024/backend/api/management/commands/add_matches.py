from datetime import datetime, timedelta
import logging
from django.core.management.base import BaseCommand
import os
import json
from django.conf import settings
from tqdm import tqdm
import random
from api.models import MatchInfo

# logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')

class Command(BaseCommand):
    help = "Command to add 10 random matches to the database"

    def handle(self, *args, **options):
        filename = 'mapping_people.json'
        data_file = os.path.join(settings.BASE_DIR, 'data', filename)

        if not os.path.exists(data_file):
            logging.error(f"File {data_file} not found.")
            return
        
        with open(data_file, 'r') as file:
            file_content = file.read()
            if not file_content.strip():
                logging.error(f"File {data_file} is empty.")
                return
            data = json.loads(file_content)
            players = data
            
        matches_to_create = 10

        for _ in tqdm(range(matches_to_create), desc="Creating matches", unit="match"):
            team_a_players=random.sample(list(players.keys()), 11)
            team_b_players=random.sample(list(players.keys()), 11)
            team_a_players = [{player: players[player]} for player in team_a_players]
            team_b_players = [{player: players[player]} for player in team_b_players]

            MatchInfo.objects.create(
                date=(datetime.now() + timedelta(days=random.randint(1, 10))).date(),
                match_type=random.choice(["test", "odi", "t20"]),
                overs=random.choice([20, 50]),
                season="2021",
                venue="Wankhede Stadium",
                team_a="Team A",
                team_a_players=json.dumps(team_a_players),
                team_b="Team B",
                team_b_players=json.dumps(team_b_players),
            )
        
        