# Generated by Django 5.1.3 on 2024-12-02 17:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0022_merge_20241202_1758'),
    ]

    operations = [
        migrations.AddField(
            model_name='matchinfo',
            name='inference_row',
            field=models.JSONField(blank=True, default=None, null=True),
        ),
        migrations.RemoveField(
            model_name='matchinfo',
            name='team_a_players',
        ),
        migrations.RemoveField(
            model_name='matchinfo',
            name='team_b_players',
        ),
        migrations.AddField(
            model_name='matchinfo',
            name='team_a_players',
            field=models.ManyToManyField(blank=True, related_name='team_a_matches', to='api.player'),
        ),
        migrations.AddField(
            model_name='matchinfo',
            name='team_b_players',
            field=models.ManyToManyField(blank=True, related_name='team_b_matches', to='api.player'),
        ),
    ]
