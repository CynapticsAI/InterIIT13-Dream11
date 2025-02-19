# Generated by Django 5.1.3 on 2024-11-30 12:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0011_alter_chat_options_remove_chat_receiver_message_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='matchinfo',
            name='amount_to_be_paid',
            field=models.IntegerField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='matchinfo',
            name='first_prize',
            field=models.CharField(blank=True, default=None, max_length=225, null=True),
        ),
        migrations.AddField(
            model_name='matchinfo',
            name='prize_pool',
            field=models.CharField(blank=True, default=None, max_length=225, null=True),
        ),
        migrations.AddField(
            model_name='matchinfo',
            name='teama_spots_left',
            field=models.IntegerField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='matchinfo',
            name='teamb_spots_left',
            field=models.IntegerField(blank=True, default=None, null=True),
        ),
    ]
