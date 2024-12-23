from rest_framework import generics
from .serializers import MatchSerializer, ChatSerializer, MessageSerializer,PlayerSerializer,MatchPostSerializer
from .models import MatchInfo, Chat, Message, Player, Team
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.response import Response
from datetime import datetime
from rest_framework import status,serializers
import pandas as pd
from datetime import datetime
import cricketstats as cks
from django.http import JsonResponse
from django.conf import settings
import os
from pathlib import Path
from django.db.models import Q
import numpy as np
from .inference import get_response
import json
from django.core.serializers.json import DjangoJSONEncoder
from .inference import get_response 
from rest_framework.pagination import LimitOffsetPagination
from .ml_model import predict_players
class InfiniteScrollPagination(LimitOffsetPagination):
    default_limit = 10  # Number of players per request
    

class MatchList(generics.ListCreateAPIView):
    serializer_class = MatchSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['team_a', 'team_b', 'city', 'match_type', 'date','id']
    ordering_fields = ['date']
    ordering = ['date']

    def get_queryset(self):
        queryset = MatchInfo.objects.all()
        limit = self.request.query_params.get('limit')
        if limit is not None and limit.isdigit():
            queryset = queryset[:int(limit)]
        return queryset


class ChatList(generics.ListCreateAPIView):
    serializer_class = ChatSerializer

    def get_queryset(self):
        match_id = self.request.query_params.get('match_id', None)
        if match_id is not None:
            return Chat.objects.filter(match_id=match_id).order_by('timestamp')
        return Chat.objects.none()



class ChatDetail(generics.RetrieveAPIView):
    queryset = Chat.objects.all()
    lookup_field = 'match'
    serializer_class = ChatSerializer


class GetPlayerReport(APIView):
    def get(self, request, player_id):
        try:
            # Construct the file path
            file_path = os.path.join(settings.BASE_DIR, 'data', 'players', 'players', f'{player_id}.json')
            
            # Read the JSON file
            with open(file_path, 'r') as file:
                player_data = json.load(file)
            
            # Transform played against data with player names
            played_against_with_names = {}
            for opponent_id, stats in player_data.get('played_against', {}).items():
                try:
                    opponent = Player.objects.get(identifier=opponent_id)
                    played_against_with_names[opponent.name] = {
                        'Runs': stats.get('runs', 0),
                        'Balls': stats.get('balls', 0),
                        'Wickets': stats.get('wickets', 0)
                    }
                except Player.DoesNotExist:
                    # Fallback to ID if player not found
                    played_against_with_names[opponent_id] = stats
            
            # Comprehensive performance transformation
            transformed_data = {
                # Personal Performance Metrics
                'Performance': {
                    'Fifties': player_data.get('fifties', 0),
                    'Thirties': player_data.get('thirties', 0),
                    'Centuries': player_data.get('centuries', 0),
                    'Maiden Overs': player_data.get('maidens', 0),
                    'Hattricks': player_data.get('hattricks', 0),
                },

                # Played Against Details
                'Played Against': played_against_with_names,

                # Club Performance Metrics
                'Club Performance': {
                    'ODI': {
                        'Lifetime Runs': player_data.get('club_ODI_lifetime_runs', 0),
                        'Lifetime Wickets': player_data.get('club_ODI_lifetime_wickets', 0),
                        'Previous 3 Runs': player_data.get('club_ODI_previous3_runs', []),
                    },
                    'Test': {
                        'Lifetime Runs': player_data.get('club_Test_lifetime_runs', 0),
                        'Lifetime Wickets': player_data.get('club_Test_lifetime_wickets', 0),
                        'Previous 3 Runs': player_data.get('club_Test_previous3_runs', []),
                    },
                    'T20': {
                        'Lifetime Runs': player_data.get('club_T20_lifetime_runs', 0),
                        'Lifetime Wickets': player_data.get('club_T20_lifetime_wickets', 0),
                        'Previous 3 Runs': player_data.get('club_T20_previous3_runs', []),
                    }
                },

                # International Performance Metrics
                'International Performance': {
                    'ODI': {
                        'Lifetime Runs': player_data.get('international_ODI_lifetime_runs', 0),
                        'Lifetime Wickets': player_data.get('international_ODI_lifetime_wickets', 0),
                        'Previous 3 Runs': player_data.get('international_ODI_previous3_runs', []),
                    },
                    'Test': {
                        'Lifetime Runs': player_data.get('international_Test_lifetime_runs', 0),
                        'Lifetime Wickets': player_data.get('international_Test_lifetime_wickets', 0),
                        'Previous 3 Runs': player_data.get('international_Test_previous3_runs', []),
                    },
                    'T20': {
                        'Lifetime Runs': player_data.get('international_T20_lifetime_runs', 0),
                        'Lifetime Wickets': player_data.get('international_T20_lifetime_wickets', 0),
                        'Previous 3 Runs': player_data.get('international_T20_previous3_runs', []),
                    }
                }
            }
            
            return Response(transformed_data, status=status.HTTP_200_OK)
        
        except FileNotFoundError:
            return Response({'error': 'Player data not found'}, status=status.HTTP_404_NOT_FOUND)
        except json.JSONDecodeError:
            return Response({'error': 'Invalid JSON file'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PredictPlayers(APIView):
    def post(self, request):
        data = request.data
        match_id = data.get('match_id')

        try:
            match = MatchInfo.objects.get(id=match_id)
            inference_row = match.inference_row
            inference_row = json.loads(inference_row)
            
            response = predict_players(inference_row)
            print(response)

            players = Player.objects.filter(name__in=response)

            serialized_players = PlayerSerializer(players, many=True).data

            return Response({
                'predicted_players': serialized_players
            }, status=status.HTTP_200_OK)

        except MatchInfo.DoesNotExist:
            return Response({'error': 'Match not found'}, status=status.HTTP_404_NOT_FOUND)

        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)


class GetAllPlayers(generics.ListAPIView):
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer
    pagination_class = InfiniteScrollPagination

class MatchCreateAPIView(generics.CreateAPIView):
    queryset = MatchInfo.objects.all()
    serializer_class = MatchPostSerializer

    def perform_create(self, serializer):
        # Save the instance, which also handles player relationships
        instance = serializer.save()

        # Fetch player names for inference
        team_a_player_names = list(instance.team_a_players.values_list('name', flat=True))
        team_b_player_names = list(instance.team_b_players.values_list('name', flat=True))

        try:
            # Call inference function
            inference_list = get_response(
                team_a_player_names,
                team_b_player_names,
                instance.match_type,
                instance.date
            )

            # Update inference_row
            instance.inference_row = json.dumps(inference_list, cls=DjangoJSONEncoder)
            instance.save()
        except Exception as e:
            print(f"Error updating inference_row in API call: {e}")
            raise serializers.ValidationError("Failed to update inference_row.")
