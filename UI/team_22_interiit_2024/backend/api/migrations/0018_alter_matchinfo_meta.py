# Generated by Django 5.1.3 on 2024-12-01 13:32

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0017_alter_matchinfo_balls_per_over_alter_matchinfo_city_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='matchinfo',
            name='meta',
            field=models.OneToOneField(blank=True, default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to='api.metadata'),
        ),
    ]
