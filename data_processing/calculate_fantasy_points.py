"""Rules for calculating the fantasy points"""
rules = {
    'T20': {
        "announced": 4,
        "run": 1,
        "boundary_bonus": 1,
        "six_bonus": 2,
        "30_run_bonus": 4,
        "half_century_bonus": 8,
        "century_bonus": 16,
        "dismissal_for_duck": -2,
        "wicket": 25,
        "lbw/bowled_bonus": 8,
        "3_wicket_haul": 4,
        "4_wicket_haul": 8,
        "5_wicket_haul": 16,
        "maiden": 12,
        "catch": 8,
        "3_catch_bonus": 4,
        "runout(DirectHit/Stumping)": 12,
        "runout(Catcher/Thrower)": 6,
        "min_overs_to_be_bowled_for_economy_points": 2,
        "economy_points": {
            "<5": 6,
            ">=5 and <=5.99": 4,
            ">=6 and <=7": 2,
            ">=10 and <=11": -2,
            ">=11.01 and <=12": -4,
            ">12": -6
        },
        "min_balls_to_be_played_for_strikerate_points": 10,
        "strike_rate_points": {
            ">170": 6,
            ">=150.01 and <=170": 4,
            ">=130 and <=150": 2,
            ">=60 and <=70": -2,
            ">=50 and <=59.99": -4,
            "<50": -6
        }
    },
    'IT20': {
        "announced": 4,
        "run": 1,
        "boundary_bonus": 1,
        "six_bonus": 2,
        "30_run_bonus": 4,
        "half_century_bonus": 8,
        "century_bonus": 16,
        "dismissal_for_duck": -2,
        "wicket": 25,
        "lbw/bowled_bonus": 8,
        "3_wicket_haul": 4,
        "4_wicket_haul": 8,
        "5_wicket_haul": 16,
        "maiden": 12,
        "catch": 8,
        "3_catch_bonus": 4,
        "runout(DirectHit/Stumping)": 12,
        "runout(Catcher/Thrower)": 6,
        "min_overs_to_be_bowled_for_economy_points": 2,
        "economy_points": {
            "<5": 6,
            ">=5 and <=5.99": 4,
            ">=6 and <=7": 2,
            ">=10 and <=11": -2,
            ">=11.01 and <=12": -4,
            ">12": -6
        },
        "min_balls_to_be_played_for_strikerate_points": 10,
        "strike_rate_points": {
            ">170": 6,
            ">=150.01 and <=170": 4,
            ">=130 and <=150": 2,
            ">=60 and <=70": -2,
            ">=50 and <=59.99": -4,
            "<50": -6
        }
    },


    'Test': {
        "announced": 4,
        "run": 1,
        "boundary_bonus": 1,
        "six_bonus": 2,
        "30_run_bonus": 0,
        "half_century_bonus": 4,
        "century_bonus": 8,
        "dismissal_for_duck": -4,
        "wicket": 16,
        "lbw/bowled_bonus": 8,
        "4_wicket_haul": 4,
        "3_wicket_haul": 4,
        "5_wicket_haul": 8,
        "maiden": 0,
        "catch": 8,
        "3_catch_bonus": 0,
        "runout(DirectHit/Stumping)": 12,
        "runout(Catcher/Thrower)": 6,
        "min_overs_to_be_bowled_for_economy_points": 2,
        "economy_points": {
            "<5": 0,
            ">=5 and <=5.99": 0,
            ">=6 and <=7": 0,
            ">=10 and <=11": 0,
            ">=11.01 and <=12": 0,
            ">12": 0
        },
        "min_balls_to_be_played_for_strikerate_points": 1,
        "strike_rate_points": {
            ">170": 0,
            ">=150.01 and <=170": 0,
            ">=130 and <=150": 0,
            ">=60 and <=70": 0,
            ">=50 and <=59.99": 0,
            "<50": 0
        }
    },

    'MDM': {
        "announced": 4,
        "run": 1,
        "boundary_bonus": 1,
        "six_bonus": 2,
        "30_run_bonus": 0,
        "half_century_bonus": 4,
        "century_bonus": 8,
        "dismissal_for_duck": -4,
        "wicket": 16,
        "lbw/bowled_bonus": 8,
        "4_wicket_haul": 4,
        "3_wicket_haul": 4,
        "5_wicket_haul": 8,
        "maiden": 0,
        "catch": 8,
        "3_catch_bonus": 0,
        "runout(DirectHit/Stumping)": 12,
        "runout(Catcher/Thrower)": 6,
        "min_overs_to_be_bowled_for_economy_points": 2,
        "economy_points": {
            "<5": 0,
            ">=5 and <=5.99": 0,
            ">=6 and <=7": 0,
            ">=10 and <=11": 0,
            ">=11.01 and <=12": 0,
            ">12": 0
        },
        "min_balls_to_be_played_for_strikerate_points": 1,
        "strike_rate_points": {
            ">170": 0,
            ">=150.01 and <=170": 0,
            ">=130 and <=150": 0,
            ">=60 and <=70": 0,
            ">=50 and <=59.99": 0,
            "<50": 0
        }
    },


    "ODI": {
        "announced": 4,
        "run": 1,
        "boundary_bonus": 1,
        "six_bonus": 2,
        "half_century_bonus": 4,
        "30_run_bonus": 0,
        "century_bonus": 8,
        "dismissal_for_duck": -3,
        "wicket": 25,
        "lbw/bowled_bonus": 8,
        "4_wicket_haul": 4,
        "3_wicket_haul": 0,
        "5_wicket_haul": 8,
        "maiden": 4,
        "catch": 8,
        "3_catch_bonus": 4,
        "runout(DirectHit/Stumping)": 12,
        "runout(Catcher/Thrower)": 6,
        "min_overs_to_be_bowled_for_economy_points": 5,
        "economy_points": {
            "<2.5": 6,
            ">=2.5 and <=3.49": 4,
            ">=3.5 and <=4.5": 2,
            ">=7 and <=8": -2,
            ">=8.01 and <=9": -4,
            ">9": -6
        },
        "min_balls_to_be_played_for_strikerate_points": 20,
        "strike_rate_points": {
            ">140": 6,
            ">=120.01 and <=140": 4,
            ">=100 and <=120": 2,
            ">=40 and <=50": -2,
            ">=30 and <=39.99": -4,
            "<30": -6
        }
    },
    "ODM": {
        "announced": 4,
        "run": 1,
        "boundary_bonus": 1,
        "six_bonus": 2,
        "half_century_bonus": 4,
        "30_run_bonus": 0,
        "century_bonus": 8,
        "dismissal_for_duck": -3,
        "wicket": 25,
        "lbw/bowled_bonus": 8,
        "4_wicket_haul": 4,
        "3_wicket_haul": 0,
        "5_wicket_haul": 8,
        "maiden": 4,
        "catch": 8,
        "3_catch_bonus": 4,
        "runout(DirectHit/Stumping)": 12,
        "runout(Catcher/Thrower)": 6,
        "min_overs_to_be_bowled_for_economy_points": 5,
        "economy_points": {
            "<2.5": 6,
            ">=2.5 and <=3.49": 4,
            ">=3.5 and <=4.5": 2,
            ">=7 and <=8": -2,
            ">=8.01 and <=9": -4,
            ">9": -6
        },
        "min_balls_to_be_played_for_strikerate_points": 20,
        "strike_rate_points": {
            ">140": 6,
            ">=120.01 and <=140": 4,
            ">=100 and <=120": 2,
            ">=40 and <=50": -2,
            ">=30 and <=39.99": -4,
            "<30": -6
        }
    }
}


def calc_fantasy_points(df, rules):
    """
    Function to calculate the fantasy points using the dataframe consisting of all the data of
    one innings of match for each player like sixes , fours , runs , wickets(caught , bowled) etc. 
    """
    for index, row in df.iterrows():
        points = 0
        points += row['runs']*rules['run']
        points += row['4s']*rules['boundary_bonus']
        points += row['6s']*rules['six_bonus']
        if row['runs'] >= 100:
            points += rules['century_bonus']
        elif row['runs'] >= 50:
            points += rules['half_century_bonus']
        elif row['runs'] >= 30:
            points += rules['30_run_bonus']
        points += row['ducks']*rules['dismissal_for_duck']

        points += row['wickets'] * rules['wicket']
        points += row['lbw/bowled']*rules['lbw/bowled_bonus']
        if row["wickets"] >= 5:
            points += rules["5_wicket_haul"]
        elif row["wickets"] == 4:
            points += rules["4_wicket_haul"]
        elif row["wickets"] == 3:
            points += rules["3_wicket_haul"]
        points += row["maidens"]*rules["maiden"]
        points += row["catches"]*rules["catch"]
        if row["catches"] >= 3:
            points += rules["3_catch_bonus"]

        points += row['runout(direct)']*rules['runout(DirectHit/Stumping)']
        points += row['runout(indirect)']*rules['runout(Catcher/Thrower)']

        if row['overs'] >= rules['min_overs_to_be_bowled_for_economy_points']:
            eco = row['runs gave']*1.0/row['overs']
            for x, y in rules["economy_points"].items():
                l = x.split(" and ")
                if len(l) == 1:
                    if eval(str(eco)+l[0]):
                        points += y
                        break
                else:
                    if eval(str(eco)+l[0]) and eval(str(eco)+l[1]):
                        points += y
                        break

        if row['balls played'] >= rules['min_balls_to_be_played_for_strikerate_points']:
            str_rate = (row['runs']*100.0)/row['balls played']
            for x, y in rules["strike_rate_points"].items():
                l = x.split(" and ")
                if len(l) == 1:
                    if eval(str(str_rate)+l[0]):
                        points += y
                        break
                else:
                    if eval(str(str_rate)+l[0]) and eval(str(str_rate)+l[1]):
                        points += y
                        break

        df.loc[index]['fantasy points'] += points
    return df
