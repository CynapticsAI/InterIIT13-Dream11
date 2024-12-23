from django.urls import path
from .views import MatchList, ChatList, ChatDetail,GetPlayerReport,ChatList,PredictPlayers,GetAllPlayers , MatchCreateAPIView


urlpatterns = [
    path('matches/', MatchList.as_view(), name='match-list'),
    path('get-player-report/<str:player_id>/',GetPlayerReport.as_view()),
    path('get-match-related-chats/',ChatList.as_view()),
    path('predict-players/',PredictPlayers.as_view()),
    path('players/',GetAllPlayers.as_view(), name='get-all-players'),
    path('add-match/',MatchCreateAPIView.as_view()),
]
