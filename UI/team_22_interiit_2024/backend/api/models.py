from django.db import models
from django.utils import timezone
import random
from .inference import get_response
from django.db.models.signals import post_save,pre_save
from django.dispatch import receiver
from django.core.serializers.json import DjangoJSONEncoder
import json



# Teams and players
class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Team"
        verbose_name_plural = "Teams"
        ordering = ['name']

# Information about the match
class MatchInfo(models.Model):
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]
    MATCH_TYPE_CHOICES = [
        ('test', 'Test'),
        ('odi', 'ODI'),
        ('t20', 'T20'),
    ]
    city = models.CharField(max_length=100,null=True,blank=True,default=None)
    date = models.DateField(null=True,blank=True,default=None)
    match_type = models.CharField(max_length=10, choices=MATCH_TYPE_CHOICES,null=True,blank=True,default=None)
    
    team_a = models.CharField(max_length=225,null=True,blank=True,default=None)
    team_a_players = models.ManyToManyField('Player', related_name='team_a_matches', blank=True)
    team_b = models.CharField(max_length=225,null=True,blank=True,default=None)
    team_b_players = models.ManyToManyField('Player', related_name='team_b_matches', blank=True)

    inference_row = models.JSONField(null=True,blank=True,default=None)
    def random_prize_pool():
        return f"${random.randint(1000, 10000)}"
    def random_first_prize():
        return f"${random.randint(500, 5000)}"
    def random_amount_to_be_paid():
        return random.randint(40, 60)
    def random_spots_left():
        return random.randint(1, 11)

    prize_pool = models.CharField(max_length=225, default=random_prize_pool,null=True,blank=True)
    first_prize = models.CharField(max_length=225, default=random_first_prize,null=True,blank=True)
    amount_to_be_paid = models.IntegerField(default=random_amount_to_be_paid,null=True,blank=True)
    teama_spots_left = models.IntegerField(default=random_spots_left,null=True,blank=True)
    teamb_spots_left = models.IntegerField(default=random_spots_left,null=True,blank=True)
    # spots_left = models.IntegerField(default=random_spots_left,null=True,blank=True)

    def __str__(self):
        return f"Match {self.match_type} on {self.date}"

    class Meta:
        verbose_name = "Match Info"
        verbose_name_plural = "Match Info"
        ordering = ['-date']

class Player(models.Model):
    name = models.CharField(max_length=100)
    gender = models.CharField(max_length=10, null=True, blank=True,default=None)
    role = models.CharField(max_length=50, null=True, blank=True,default=None)
    identifier = models.CharField(max_length=10, null=True, blank=True, unique=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Player"
        verbose_name_plural = "Players"
        ordering = ['name']

class Chat(models.Model):
    match = models.ForeignKey('MatchInfo', on_delete=models.CASCADE, related_name="chats")
    message = models.TextField()
    is_ai = models.BooleanField(default=False)  
    is_user = models.BooleanField(default=False)
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        message_origin = "User" if self.is_user else "AI"
        return f"{message_origin} message for match {self.match_id} at {self.timestamp}"

    class Meta:
        ordering = ['timestamp']

class Message(models.Model):
    chat = models.ForeignKey("Chat",  on_delete=models.CASCADE, related_name="messages")
    text = models.TextField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message {self.id}"

@receiver(post_save, sender=MatchInfo)
def update_inference_row(sender, instance, **kwargs):
    try:
        # team_a_players_ids = list(instance.team_a_players.values_list('identifier', flat=True))
        team_a_players = list(instance.team_a_players.values_list('name', flat=True))
        # team_b_players_ids = list(instance.team_b_players.values_list('identifier', flat=True))
        team_b_players = list(instance.team_b_players.values_list('name', flat=True))
        print('team_a_players',team_a_players,team_b_players)
        # inference_list = get_response(team_a_players,team_b_players,team_a_players_ids,team_b_players_ids, instance.match_type, instance.date)

        inference_list = get_response(team_a_players,team_b_players,instance.match_type, instance.date)
        
        instance.inference_row = inference_list
        
        MatchInfo.objects.filter(pk=instance.pk).update(
            inference_row=json.dumps(inference_list, cls=DjangoJSONEncoder)
        )
    except Exception as e:
        print(f"Error updating inference row: {e}")

