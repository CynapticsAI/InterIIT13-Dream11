import os
import json
import random
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Player
from tqdm import tqdm

class Command(BaseCommand):
    help = "Command to upload player data from JSON to the server with specified roles"

    def handle(self, *args, **options):
        # Predefined list of player roles
        PLAYER_ROLES = ["WK", "BAT", "BOWL", "AR", "BAT", "BOWL", "AR", "BOWL", "BAT"]

        # Path for JSON file
        json_file_path = os.path.join(settings.BASE_DIR, 'data', 'player_id.json')
        
        # Check if file exists
        if not os.path.exists(json_file_path):
            self.stdout.write(self.style.ERROR(f"JSON file {json_file_path} not found."))
            return

        # Load player IDs from JSON
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            player_ids = json.load(json_file)

        players_to_create = []
        players_to_update = []

        # Shuffle roles to distribute them randomly
        roles = PLAYER_ROLES * (len(player_ids) // len(PLAYER_ROLES) + 1)
        random.shuffle(roles)

        for idx, (name, player_id) in enumerate(tqdm(player_ids.items(), desc="Processing players", unit="player")):
            # Generate a unique identifier
            identifier =  player_id

            player_data = {
                "identifier": identifier,
                "name": name,
                "role": roles[idx]  # Assign roles sequentially after shuffling
            }

            try:
                # Try to get existing player
                player = Player.objects.get(identifier=identifier)
                
                # Update existing player
                for key, value in player_data.items():
                    setattr(player, key, value)
                players_to_update.append(player)
            except Player.DoesNotExist:
                # Create new player
                players_to_create.append(Player(**player_data))

        # Bulk create new players
        if players_to_create:
            Player.objects.bulk_create(players_to_create)
            self.stdout.write(self.style.SUCCESS(f"Created {len(players_to_create)} players"))

        # Bulk update existing players
        if players_to_update:
            Player.objects.bulk_update(players_to_update, [
                "name", "role"
            ])
            self.stdout.write(self.style.SUCCESS(f"Updated {len(players_to_update)} players"))

        self.stdout.write(self.style.SUCCESS("Player data upload complete."))