from rest_framework import serializers
from .models import MatchInfo, Team, Player, Message, Chat


class PlayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Player
        fields = ["id","identifier","name","role"]

class MatchSerializer(serializers.ModelSerializer):
    team_a_players = PlayerSerializer(many=True)
    team_b_players = PlayerSerializer(many=True)
    class Meta:
        model = MatchInfo
        fields = "__all__"

class MessageSerializer(serializers.ModelSerializer):
    created_at_formatted = serializers.SerializerMethodField()

    class Meta:
        model = Message
        exclude = []
        depth = 1

    def get_created_at_formatted(self, obj:Message):
        return obj.created_at.strftime("%d-%m-%Y %H:%M:%S")

class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = "__all__"


class MatchPostSerializer(serializers.ModelSerializer):
    team_a_players = serializers.ListField(
        child=serializers.CharField(),
        write_only=True
    )
    team_b_players = serializers.ListField(
        child=serializers.CharField(),
        write_only=True
    )

    class Meta:
        model = MatchInfo
        fields = "__all__"

    def create(self, validated_data):
        # Extract player names
        team_a_player_names = validated_data.pop('team_a_players', [])
        team_b_player_names = validated_data.pop('team_b_players', [])
        print(team_b_player_names,team_a_player_names)
        # Fetch player instances
        team_a_players = list(Player.objects.filter(name__in=team_a_player_names))
        team_b_players = list(Player.objects.filter(name__in=team_b_player_names))
        print(team_a_players,team_b_players)
        print(team_a_players,team_b_players)
        if len(team_a_players) != len(team_a_player_names):
            raise serializers.ValidationError("Some players in team_a not found.")
        if len(team_b_players) != len(team_b_player_names):
            raise serializers.ValidationError("Some players in team_b not found.")

        # Create MatchInfo instance
        match_info = super().create(validated_data)

        # Add players to ManyToMany fields
        match_info.team_a_players.set(team_a_players)
        match_info.team_b_players.set(team_b_players)

        return match_info
