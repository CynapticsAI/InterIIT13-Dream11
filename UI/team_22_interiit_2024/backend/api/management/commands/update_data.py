import os
import django
import random
from django.core.management.base import BaseCommand
from django.utils import timezone
from api.inference import get_response
from django.core.serializers.json import DjangoJSONEncoder
import json
# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
django.setup()

from api.models import MatchInfo, Player  # Replace with your actual app and model names
class Command(BaseCommand):
    help = 'Populate MatchInfo model with random data and generate inference rows'

    # International cricket teams
    INTERNATIONAL_TEAMS = [
        'India', 'Australia', 'England', 'Pakistan', 'South Africa', 
        'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'Afghanistan',
        'Ireland', 'Netherlands', 'Zimbabwe', 'Scotland', 'UAE'
    ]

    def handle(self, *args, **options):
        # Clear existing MatchInfo entries (optional)
        MatchInfo.objects.all().delete()

        # Generate multiple matches
        num_matches = 5  # You can adjust this number
        for _ in range(num_matches):
            # Randomly select two teams
            team_a, team_b = random.sample(self.INTERNATIONAL_TEAMS, 2)

            # Get all players
            all_players = list(Player.objects.all())

            # Ensure we have enough players
            if len(all_players) < 22:
                self.stdout.write(self.style.ERROR('Not enough players in the database!'))
                return

            # Randomly select 11 players for each team
            team_a_players = random.sample(all_players, 11)
            remaining_players = [p for p in all_players if p not in team_a_players]
            team_b_players = random.sample(remaining_players, 11)

            # Create match
            match = MatchInfo.objects.create(
                city=self.generate_random_city(),
                date=self.generate_random_date(),
                match_type=random.choice(['test', 'odi', 't20']),
                team_a=team_a,
                team_b=team_b,
                prize_pool=f"${random.randint(1000, 10000)}",
                first_prize=f"${random.randint(500, 5000)}",
                amount_to_be_paid=random.randint(100, 1000),
                teama_spots_left=random.randint(1, 11),
                teamb_spots_left=random.randint(1, 11)
            )

            # Add players to the match
            match.team_a_players.set(team_a_players)
            match.team_b_players.set(team_b_players)

            # Generate inference row
            try:
                # Extract player names
                team_a_players_names = list(match.team_a_players.values_list('name', flat=True))
                team_b_players_names = list(match.team_b_players.values_list('name', flat=True))

                # Generate inference data
                inference_list = get_response(
                    team_a_players_names, 
                    team_b_players_names, 
                    match.match_type, 
                    match.date
                )

                # Update the match with inference data
                match.inference_row = json.dumps(inference_list, cls=DjangoJSONEncoder)
                match.save()

                self.stdout.write(self.style.SUCCESS(f'Created match: {match} with inference data'))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error updating inference row: {e}"))

    def generate_random_city(self):
        cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
            'Kolkata', 'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow'
        ]
        return random.choice(cities)

    def generate_random_date(self):
        # Generate a random date within the last 2 years
        days_back = random.randint(0, 730)
        return timezone.now().date() - timezone.timedelta(days=days_back)