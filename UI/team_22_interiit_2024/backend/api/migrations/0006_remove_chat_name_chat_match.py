# Generated by Django 5.1.3 on 2024-11-12 15:42

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0005_matchinfo_team_a_players_matchinfo_team_b_players_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='chat',
            name='name',
        ),
        migrations.AddField(
            model_name='chat',
            name='match',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.CASCADE, related_name='chats', to='api.matchinfo'),
            preserve_default=False,
        ),
    ]
