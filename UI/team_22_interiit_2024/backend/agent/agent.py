import fal_client
import os
from termcolor import colored
import json
from py_expression_eval import Parser
from gtts import gTTS
from io import BytesIO
from pathlib import Path
from sqlalchemy import create_engine
import sqlite3
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyAHljbMCfSFRLNPCiUwlBWJq1HgIdSwF4c"
WEATHER_API_KEY = "ZWUJRENGDYL5EZ84H58KTNDRL"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

parser = Parser()

AGENT_PROMPT = '''
    Respond with a python dictionary to the final content of user by abolutely adherring to the format below. Example = Show your "Action" by "Action: (Toolname)" followed by "Action Input: (Exact Input)".
    Dictionary format: {"Thought": "Your exact thought", "Action": "The action you are taking", "Action Input": "The inputs of the action you are taking"}

    You have access to the following tools:-

    1. Calculator: Useful for when you need to answer questions about math. Use python code, eg: 2 + 2
    2. SearchDB: Useful for searching into Cricket Data when you need to search into the database to find any data related to cricket. Use python code, eg: cricket bowler has highest wickets
    3. Response To Human: When you need to respond to the human you are talking to.
    
    You will receive a message from the human, then you should start a loop and do one of three things

    Option 1: You use a tool to answer the question.
        For this, you should use the following format:
        Thought: you should always think about what to do
        Action: the action to take, should be one of [Calculator, SearchDB]
        Action Input: the input to the action, to be sent to the tool
    
    After this, the human will respond with an observation, and you will continue.
    
    Option 2: You respond to the human.
        For this, you should use the following format:
        Action: Response To Human
        Action Input: your response to the human, summarizing what you did and what you learned

    Option 3: You need to end the conversation after finishing your task.
        For this, you should use the following format:
        Action: Break
        Action Input: None
    
    Examples:- 
        If the user says "Tell me the name of player with most wickets", {"Thought": "Need to access database using queries to find player with most wickets.", "Action":  "SearchDB", "Action Input": "player with most wickets"}
        If the user says "Tell me data of player Kohli", {"Thought": "Need to access database using queries and search someone Kohli.", "Action":  "SearchDB", "Action Input": "data of player with similar name as Kohli"}
        If the user says "Tell me the sum of 20 and 5",  {"Thought": "Need to access Calculator.", "Action":  "Calculator", "Action Input": "20 + 5"}
        If you want to talk normally if the user says "Hello!", {"Thought": "Need to greet.", "Action":  "Response To Human", "Action Input": "greet"}
        If user's first query has answered, {"Thought": "Need to end.", "Action":  "Break", "Action Input": "None"}
        If final two messages are the same, {"Thought": "Need to break loop.", "Action":  "Break", "Action Input": "None"}
    Begin!
'''

SQL_PROMPT = '''Create me an exact SQL query as a string for the prompt given at the end. Example: If the user says "Fetch me all players", your response should be SELECT * FROM api_player; without any extra letters
                        Default entry type for each table shown after an arrow (->) = (index, column_name, data_type, nullable, default_value, some_flag)
                        The following tables exist in the database along with their columns described in SQLite:
                            api_delivery -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'batter', 'varchar(100)', 1, None, 0), (2, 'bowler', 'varchar(100)', 1, None, 0), (3, 'non_striker', 'varchar(100)', 1, None, 0), (4, 'batter_runs', 'INTEGER', 1, None, 0), (5, 'extras_runs', 'INTEGER', 1, None, 0), (6, 'total_runs', 'INTEGER', 1, None, 0), (7, 'over_id', 'bigint', 1, None, 0)]
                            api_wicket -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'kind', 'varchar(100)', 1, None, 0), (2, 'player_out', 'varchar(100)', 1, None, 0), (3, 'delivery_id', 'bigint', 1, None, 0), (4, 'fielder_id', 'bigint', 0, None, 0)]
                            api_extra -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'wides', 'INTEGER', 0, None, 0), (2, 'legbyes', 'INTEGER', 0, None, 0), (3, 'delivery_id', 'bigint', 1, None, 0)]
                            api_inning -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'team', 'varchar(100)', 1, None, 0), (2, 'match_info_id', 'bigint', 1, None, 0)n]
                            api_matchinfo -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'balls_per_over', 'INTEGER', 1, None, 0), (2, 'city', 'varchar(100)', 1, None, 0), (3, 'date', 'date', 1, None, 0), (4, 'gender', 'varchar(10)', 1, None, 0), (5, 'match_type', 'varchar(10)', 1, None, 0), (6, 'match_type_number', 'INTEGER', 1, None, 0), (7, 'overs', 'INTEGER', 1, None, 0), (8, 'season', 'varchar(20)', 1, None, 0), (9, 'team_type', 'varchar(50)', 1, None, 0), (10, 'venue', 'varchar(100)', 1, None, 0), (11, 'player_of_match', 'varchar(100)', 1, None, 0), (12, 'toss_decision', 'varchar(10)', 1, None, 0), (13, 'toss_winner', 'varchar(100)', 1, None, 0), (14, 'target_runs', 'INTEGER', 0, None, 0), (15, 'target_overs', 'INTEGER', 0, None, 0), (16, 'meta_id', 'bigint', 1, None, 0), (17, 'team_a_id', 'bigint', 0, None, 0), (18, 'team_b_id', 'bigint', 0, None, 0)]
                            api_metadata -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'data_version', 'varchar(10)', 1, None, 0), (2, 'created', 'date', 1, None, 0), (3, 'revision', 'INTEGER', 1, None, 0)]
                            api_official -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'match_referee', 'varchar(100)', 0, None, 0), (2, 'tv_umpire', 'varchar(100)', 0, None, 0), (3, 'umpire', 'varchar(100)', 1, None, 0), (4, 'match_info_id', 'bigint', 1, None, 0)]
                            api_outcome -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'runs', 'INTEGER', 0, None, 0), (2, 'wickets', 'INTEGER', 0, None, 0), (3, 'winner', 'varchar(100)', 1, None, 0), (4, 'match_info_id', 'bigint', 1, None, 0)]
                            api_over -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'over_number', 'INTEGER', 1, None, 0), (2, 'inning_id', 'bigint', 1, None, 0)]
                            api_player -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'name', 'varchar(100)', 1, None, 0), (2, 'unique_name', 'varchar(100)', 1, None, 0), (3, 'identifier', 'varchar(10)', 0, None, 0), (4, 'key_bcci', 'varchar(100)', 0, None, 0), (5, 'key_bcci_2', 'varchar(100)', 0, None, 0), (6, 'key_bigbash', 'varchar(100)', 0, None, 0), (7, 'key_cricbuzz', 'varchar(100)', 0, None, 0), (8, 'key_cricheroes', 'varchar(100)', 0, None, 0), (9, 'key_crichq', 'varchar(100)', 0, None, 0), (10, 'key_cricinfo', 'varchar(100)', 0, None, 0), (11, 'key_cricinfo_2', 'varchar(100)', 0, None, 0), (12, 'key_cricingif', 'varchar(100)', 0, None, 0), (13, 'key_cricketarchive', 'varchar(100)', 0, None, 0), (14, 'key_cricketarchive_2', 'varchar(100)', 0, None, 0), (15, 'key_cricketworld', 'varchar(100)', 0, None, 0), (16, 'key_nvplay', 'varchar(100)', 0, None, 0), (17, 'key_nvplay_2', 'varchar(100)', 0, None, 0), (18, 'key_opta', 'varchar(100)', 0, None, 0), (19, 'key_opta_2', 'varchar(100)', 0, None, 0), (20, 'key_pulse', 'varchar(100)', 0, None, 0), (21, 'key_pulse_2', 'varchar(100)', 0, None, 0)]
                            api_powerplay -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'from_over', 'decimal', 1, None, 0), (2, 'to', 'decimal', 1, None, 0), (3, 'powerplay_type', 'varchar(10)', 1, None, 0), (4, 'match_info_id', 'bigint', 1, None, 0)]
                            api_team -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'name', 'varchar(100)', 1, None, 0)]
                            api_team_players -> [(0, 'id', 'INTEGER', 1, None, 1), (1, 'team_id', 'bigint', 1, None, 0), (2, 'player_id', 'bigint', 1, None, 0)]
                            --------
                            END OF TABLES

                            Below is the user prompt
                            '''

def calculator(input: str):
    return parser.parse(input).evaluate({})

def searchDB(input: str):
    conn = sqlite3.connect('db.sqlite3')
    cursor = conn.cursor()

    # output = model.generate_content(PROMPT + input)
    # print(output.candidates[0].content.parts[0].text, end = '\n')
    # data = str(output.candidates[0].content.parts[0].text)
    handler = fal_client.submit(
        "fal-ai/any-llm",
        arguments={
            "prompt": SQL_PROMPT + input,      
            "model": "openai/gpt-4o-mini"
        },
    )

    request_id = handler.request_id
    status = fal_client.status("fal-ai/any-llm", request_id, with_logs=True)
    data = fal_client.result("fal-ai/any-llm", request_id)
    data = str(data['output'])
    print(status, end = '\n')
    print(data, end = '\n')

    cursor.execute(data)
    data = cursor.fetchall()
    
    cursor.close()
    conn.close()

    print(data, end = '\n')

    return data

def respondHuman(input: str):
    handler = fal_client.submit(
        "fal-ai/any-llm",
        arguments={
            "prompt": input,      
            "model": "openai/gpt-4o-mini"
        },
    )

    request_id = handler.request_id
    status = fal_client.status("fal-ai/any-llm", request_id, with_logs=True)
    data = fal_client.result("fal-ai/any-llm", request_id)
    data = data['output']
    print(status, end = '\n')
    print(data, end = '\n')

    
    return data

# def weather(input: str):
#     return "Windy"

def agentAI(message: str):
    # Messages
    messages = [
        {"content": AGENT_PROMPT + message},
    ]

    while True:
        tool = None
        response = ''

        # output = model.generate_content(str(messages))
        # if output.candidates and output.candidates[0].content.parts:
        #     response = str(output.candidates[0].content.parts[0].text)
        #     print(response, end = '\n')
        # else:
        #     response = ''

        handler = fal_client.submit(
            "fal-ai/any-llm",
            arguments={
                "prompt": str(messages),      
                "model": "openai/gpt-4o-mini"
            },
        )

        request_id = handler.request_id
        status = fal_client.status("fal-ai/any-llm", request_id, with_logs=True)
        response = fal_client.result("fal-ai/any-llm", request_id)
        output = response['output']
        response = json.loads(output)
        print(status, end = '\n')
        print(response, end = '\n')

        # action = response.split("Action:")[1].split("\n")[0].strip()
        # action_input = response.split("Action Input:")[1].split("\n")[0].strip()
        print(type(response))
        action = response["Action"]
        print(action)
        action_input = response["Action Input"]
        print(action_input)

        if action == "Calculator":
            tool = calculator
        elif action == "SearchDB":
            tool = searchDB
        elif action == "Response To Human":
            tool = respondHuman
        elif action == "Break":
            break

        if tool:
            observation = tool(action_input)
            print(observation, end = '\n')
            messages.append({"content": observation})

agentAI("Fetch me all data of the first row of player")