import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import MatchInfo, Chat
from .agent import get_response

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    @database_sync_to_async
    def get_match_players(self, match_id):
        print('match_id',match_id)
        match = MatchInfo.objects.get(id=match_id)
        print(match)
        team_a_players = list(match.team_a_players.values_list('identifier', flat=True))
        team_b_players = list(match.team_b_players.values_list('identifier', flat=True))
        print('players in each team',team_a_players,team_b_players)
        return team_a_players + team_b_players

    @database_sync_to_async
    def save_user_message(self, match_id, message):
            print("match_id in save user message",match_id)
            match = MatchInfo.objects.get(id=match_id)
            user_chat = Chat.objects.create(
                match=match,
                message=message,
                is_user=True,  # Message sent from frontend
                is_ai=False   # Not an AI response
            )
            return user_chat.id

    @database_sync_to_async
    def save_ai_response(self, match_id, message):
            match = MatchInfo.objects.get(id=match_id)
            ai_chat = Chat.objects.create(
                match=match,
                message=message,
                is_user=False,  # Not a frontend message
                is_ai=True     # Response from AI
            )
            return ai_chat.id

    async def receive(self, text_data):
        print('text_data', text_data)
        data = json.loads(text_data)
        message = data.get("message", "")
        match_id = data.get("match_id", "")
        language = data.get("language", "")
        team_a_players = data.get('team_a_player', [])
        team_b_players = data.get('team_b_player', [])
        print('match_id', match_id)
        print('team_a_players', team_a_players)
        print('team_b_players', team_b_players)

        # Combine team_a_players and team_b_players
        all_players = team_a_players + team_b_players
        print('all_players', all_players)
        print(all_players)
        # Send message to LLM and get response
        llm_response = await self.run_agent_ai(message, all_players, language)

        # Send response back to client
        await self.send(text_data=json.dumps({
            'status': 'success',
            'message': str(llm_response)
        }))

    @database_sync_to_async
    def run_agent_ai(self, message, all_players, language):
        # Wrap the synchronous agentAI function 
        return get_response(message, all_players, language)

# Serializer and List View remain the same as in the previous example