"""
Classes to create the player object to store the player data and modify it 
with each match and to store the head to head data of all players
"""


class Player:
    player_attr = ['runs', 'wickets', 'balls played', 'overs', 'runs gave']
    player_attributes = ['runs', 'wickets',
                         'balls played', 'economy', 'strike rate']
    match_tiers = ['club', 'international']
    match_types = ['ODI', 'Test', 'T20']
    duration = ['previous3', 'lifetime']

    def __init__(self):
        self.played_against = {}
        self.fifties = 0
        self.thirties = 0
        self.centuries = 0
        self.maidens = 0
        self.hattricks = 0
        self.runs = 0
        self.wickets = 0
        for tier in self.match_tiers:
            for types in self.match_types:
                for duration_type in self.duration:
                    for attr in self.player_attr:
                        s = tier + '_' + types + '_' + duration_type + '_' + attr
                        self.set_variable(
                            s, 0.0 if duration_type == 'lifetime' else [0.0, 0.0, 0.0])

    def set_variable(self, var_name, value):
        setattr(self, var_name, value)

    def get_variable(self, var_name):
        return getattr(self, var_name, None)

    def __getitem__(self, var_name):
        return getattr(self, var_name)

    def __setitem__(self, var_name, value):
        setattr(self, var_name, value)


class Player_against:
    def __init__(self):
        self.runs = 0
        self.balls = 0
        self.wickets = 0

    def set_variable(self, var_name, value):
        setattr(self, var_name, value)

    def get_variable(self, var_name):
        return getattr(self, var_name, None)

    def __getitem__(self, var_name):
        return getattr(self, var_name)

    def __setitem__(self, var_name, value):
        setattr(self, var_name, value)
