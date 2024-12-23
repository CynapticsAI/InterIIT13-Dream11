from django.contrib import admin
from .models import ( MatchInfo,  Player, Chat, Message
)


@admin.register(MatchInfo)
class MatchInfoAdmin(admin.ModelAdmin):
    list_display = ('id','match_type', 'date', 'team_a', 'team_b')
    list_filter = ( 'match_type',)
    search_fields = ('team_a__name', 'team_b__name')


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ('id','name',  'identifier')
    search_fields = ('id','name', 'identifier')

    def teams_display(self, obj):
        return ', '.join(team.name for team in obj.teams.all())
    teams_display.short_description = 'Teams'


admin.site.register(Chat)
admin.site.register(Message)