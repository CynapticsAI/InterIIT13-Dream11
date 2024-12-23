import torch
import torch.nn as nn
import torch.utils.data as data
import math
import os
import json 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def convert_row(row):
    row[0] = 0
    row[8:30] = [0] * 22
    # print(row)
    row = np.array(row)

    match_data = torch.tensor(row[1:8], dtype=torch.float32).unsqueeze(0).expand(22, 7)

    # Process target data for each player (22 players, 15 stats)
    player_targets = []
    player_idxs = row[30:360][14::15].argsort()[::-1]

    for i in player_idxs:
        temp = torch.zeros(192)
        temp[100:115] = torch.tensor(row[30+i*15:45+i*15], dtype=torch.float32) 
        temp[115+i] = 1
        player_targets.append(temp)
    target = torch.stack(player_targets)  # Shape: [22, 192]

    # Process source data for each player (22 players, 33 stats)
    head_to_head = []
    for i in range(22):
        head_to_head_stats = row[360 + (i*33): 393 + (i*33)]
        head_to_head.append(torch.tensor(head_to_head_stats, dtype=torch.float32))

    head_to_head = torch.stack(head_to_head)  # Shape: [22, 33]

    player_data = []
    for i in range(22):
        player_stats = row[1086 + (i*60):1146 + (i*60)]
        player_data.append(torch.tensor(player_stats, dtype=torch.float32))
    player_data = torch.stack(player_data)  # Shape: [22, 60]

    # Concatenate match data with player data
    source = torch.cat([match_data, player_data, head_to_head], dim=1)  # Shape: [22, 100]
    src = torch.zeros((22, 192))
    src[:, :100] = source
    tokens = torch.cat([src, target]) # Shape: [44, 100]

    return src

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert (n_heads * self.head_dim == d_model)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape

        # Project the input embeddings into Q, K, and V
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        # Compute the weighted sum of the values
        attention_output = torch.matmul(self.dropout(attention_weights), V)

        # Concatenate heads and put them back to the original shape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        # Apply the final linear transformation
        out = self.fc_out(attention_output)
        return out


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(0.2)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, logits):
        att_logits = self.att(logits)
        adn_logits = self.bn1((logits + att_logits).permute(0, 2, 1)).permute(0, 2, 1)
        logits = self.dropout(logits + att_logits)
        logits = self.fcn(logits)
        logits = self.bn2((logits + adn_logits).permute(0, 2, 1)).permute(0, 2, 1)
        return logits


class GPT(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in  range(n_layers)])
        self.linear1 = nn.Linear(d_model, d_model)
    
    def forward(self, inputs, targets = None):
        logits = inputs

        for block in self.blocks:
            logits = block(logits)
        
        logits = self.linear1(logits)
        
        loss = None
        if targets != None:
            batch_size, sequence_length, d_model = logits.shape
            # to calculate loss for all token embeddings in a batch
            # kind of a requirement for cross_entropy
            logits_ce = logits[:,21:,115:137]
            targets_ce = targets[:,21:,115:137]
            ce_loss = F.cross_entropy(logits_ce, targets_ce)

            # input_loss = F.mse_loss(logits[:,:21, :], targets[:, :21, :])
            # print(logits[0,30,:15], targets[0, 30, :15])
            mse_loss = F.mse_loss(logits[:,21:,100:115], targets[:, 21:, 100:115])
            # mse_loss = F.mse_loss(logits, targets)
            loss = mse_loss + ce_loss  #+ input_loss
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        outputs = []
        for _ in range(max_new_tokens):

            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            probs = F.softmax(logits[:, 115:137], dim=1)
            # get the probable token based on the input probs
            # idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.argsort(probs, dim=1)[0]
            # select the index which is not yet selected
            i = 0
            while idx_next[i] in outputs:
                i += 1
            inputs = torch.cat([inputs, logits.unsqueeze(1)], dim=1)
            outputs.append(idx_next[i].item())
        return outputs


import pandas as pd
import numpy as np
import json
import pickle

match_tiers = ['club', 'international']
match_types = ['ODI', 'Test', 'T20']
duration = ['previous3', 'lifetime']
h2h_attr = ['runs', 'balls', 'wickets']
player_attributes = ['runs', 'wickets',
                     'balls played', 'economy', 'strike rate']
fantasy_points_attr = ['runs', 'wickets', '4s', '6s', 'catches', 'maidens', 'overs',
                       'runs gave', 'out', 'ducks', 'balls played', 'lbw/bowled', 'runout(indirect)', 'runout(direct)', 'fantasy points']
cols = ['date', 'ODI', 'Test', 'T20', 'venue',
        'club', 'international', 'match_state']

date = '2014'
match_tier = 'international'


def make_inference_row(team1_names, team2_names, match_type):
    """
    The function needs an input in the form of list of names for team1 and team2 players
    and it outputs the best playing 11 according to the model stored in model.pkl
    """
    values = []
    values.append(date)
    for type in match_types:
        if match_type == type:
            values.append(1)
        else:
            values.append(0)

    # venue
    values.append(0)

    # match tier
    for tier in match_tiers:
        if tier == match_tier:
            values.append(1)
        else:
            values.append(0)

    # match state
    values.append(1)

    player_data = {}
    with open('../data/processed/player_id.json', 'r') as f:
        player_ids = json.load(f)

    team1 = []
    team2 = []

    # team players
    for idx in range(0, 11):
        values.append(team1_names[idx])
        player_id = player_ids[team1_names[idx]]
        team1.append(player_id)
        with open('../data/processed/players/' + player_id + '.json') as f:
            player_data[player_id] = json.load(f)
    for idx in range(0, 11):
        values.append(team2_names[idx])
        player_id = player_ids[team2_names[idx]]
        team2.append(player_id)
        with open('../data/processed/players/' + player_id + '.json') as f:
            player_data[player_id] = json.load(f)

    # current match stats(fantasy points related)
    for i in range(0, 22):
        for attr in fantasy_points_attr:
            values.append(0)

    # head to head
    for i in range(0, 11):
        team1_player_obj = player_data[team1[i]]
        for j in range(0, 11):
            team2_player_id = team2[j]
            if team2_player_id in team1_player_obj['played_against'].keys():
                team2_player_obj = team1_player_obj['played_against'][team2_player_id]
                values.append(team2_player_obj['runs'])
                values.append(team2_player_obj['balls'])
                values.append(team2_player_obj['wickets'])
            else:
                for k in range(0, 3):
                    values.append(0)

    for i in range(0, 11):
        team2_player_obj = player_data[team2[i]]
        for j in range(0, 11):
            team1_player_id = team1[j]
            if team1_player_id in team2_player_obj['played_against'].keys():
                team1_player_obj = team2_player_obj['played_against'][team1_player_id]
                values.append(team1_player_obj['runs'])
                values.append(team1_player_obj['balls'])
                values.append(team1_player_obj['wickets'])
            else:
                for k in range(0, 3):
                    values.append(0)

    for player_id in team1:
        player = player_data[player_id]
        for tier in match_tiers:
            for type in match_types:
                for duration_type in duration:
                    for attr in player_attributes:
                        if attr == 'strike rate':
                            if duration_type == 'lifetime':
                                runs = player[tier + '_' + type +
                                              '_' + duration_type + '_runs']
                                balls = player[tier + '_' + type +
                                               '_' + duration_type + '_balls played']
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                            else:
                                runs_list = player[tier + '_' +
                                                   type + '_' + duration_type + '_runs']
                                balls_list = player[tier + '_' + type +
                                                    '_' + duration_type + '_balls played']
                                runs = sum(runs_list)
                                balls = sum(balls_list)
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                        elif attr == 'economy':
                            if duration_type == 'lifetime':
                                runs_gave = player[tier + '_' + type +
                                                   '_' + duration_type + '_runs gave']
                                overs = player[tier + '_' + type +
                                               '_' + duration_type + '_overs']
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                            else:
                                runs_gave_list = player[tier + '_' +
                                                        type + '_' + duration_type + '_runs gave']
                                overs_list = player[tier + '_' +
                                                    type + '_' + duration_type + '_overs']
                                runs_gave = sum(runs_gave_list)
                                overs = sum(overs_list)
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                        else:
                            if duration_type == 'lifetime':
                                values.append(
                                    player[tier + '_' + type + '_' + duration_type + '_' + attr])
                            else:
                                curr = player[tier + '_' + type +
                                              '_' + duration_type + '_' + attr]
                                tot = sum(curr)
                                values.append(tot)

    for player_id in team2:
        player = player_data[player_id]
        for tier in match_tiers:
            for type in match_types:
                for duration_type in duration:
                    for attr in player_attributes:
                        if attr == 'strike rate':
                            if duration_type == 'lifetime':
                                runs = player[tier + '_' + type +
                                              '_' + duration_type + '_runs']
                                balls = player[tier + '_' + type +
                                               '_' + duration_type + '_balls played']
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                            else:
                                runs_list = player[tier + '_' +
                                                   type + '_' + duration_type + '_runs']
                                balls_list = player[tier + '_' + type +
                                                    '_' + duration_type + '_balls played']
                                runs = sum(runs_list)
                                balls = sum(balls_list)
                                if balls == 0:
                                    values.append(0)
                                else:
                                    values.append(runs*100.0/balls)
                        elif attr == 'economy':
                            if duration_type == 'lifetime':
                                runs_gave = player[tier + '_' + type +
                                                   '_' + duration_type + '_runs gave']
                                overs = player[tier + '_' + type +
                                               '_' + duration_type + '_overs']
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                            else:
                                runs_gave_list = player[tier + '_' +
                                                        type + '_' + duration_type + '_runs gave']
                                overs_list = player[tier + '_' +
                                                    type + '_' + duration_type + '_overs']
                                runs_gave = sum(runs_gave_list)
                                overs = sum(overs_list)
                                if overs == 0:
                                    values.append(100)
                                else:
                                    values.append(runs_gave*1.0/overs)
                        else:
                            if duration_type == 'lifetime':
                                values.append(
                                    player[tier + '_' + type + '_' + duration_type + '_' + attr])
                            else:
                                curr = player[tier + '_' + type +
                                              '_' + duration_type + '_' + attr]
                                tot = sum(curr)
                                values.append(tot)
    return values

def convert_row(row):
    row[0] = 0
    row[8:30] = [0] * 22
    # print(row)
    row = np.array(row)

    match_data = torch.tensor(row[1:8], dtype=torch.float32).unsqueeze(0).expand(22, 7)

    # Process target data for each player (22 players, 15 stats)
    player_targets = []
    player_idxs = row[30:360][14::15].argsort()[::-1]

    for i in player_idxs:
        temp = torch.zeros(192)
        temp[100:115] = torch.tensor(row[30+i*15:45+i*15], dtype=torch.float32) 
        temp[115+i] = 1
        player_targets.append(temp)
    target = torch.stack(player_targets)  # Shape: [22, 192]

    # Process source data for each player (22 players, 33 stats)
    head_to_head = []
    for i in range(22):
        head_to_head_stats = row[360 + (i*33): 393 + (i*33)]
        head_to_head.append(torch.tensor(head_to_head_stats, dtype=torch.float32))

    head_to_head = torch.stack(head_to_head)  # Shape: [22, 33]

    player_data = []
    for i in range(22):
        player_stats = row[1086 + (i*60):1146 + (i*60)]
        player_data.append(torch.tensor(player_stats, dtype=torch.float32))
    player_data = torch.stack(player_data)  # Shape: [22, 60]

    # Concatenate match data with player data
    source = torch.cat([match_data, player_data, head_to_head], dim=1)  # Shape: [22, 100]
    src = torch.zeros((22, 192))
    src[:, :100] = source
    tokens = torch.cat([src, target]) # Shape: [44, 100]
    return src



if __name__ == "__main__":
    with open(os.path.join("../model_artifacts/model_config.json"), "r") as f:
        data = json.load(f)
    running_mean = data["running_mean"]
    running_std = data["running_std"]
    d_model = data["d_model"]
    n_heads = data["n_heads"]
    n_layers = data["n_layers"]

    model = GPT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to("mps")
    model.load_state_dict(torch.load("../model_artifacts/final_model.pt", map_location="mps"))
    model.eval()
    team1_names = ['NG Jones', 'C Dougherty', 'LT Nelson', 'Mansoor Amjad', 'J Holmes',
               'NL Smith', 'GJ Thompson', 'PA Eakin', 'GJ McCarter', 'GE Kidd', 'PS Eaglestone']
    team2_names = ['DI Joyce', 'FP McAllister', 'J Anderson', 'Simi Singh', 'AD Poynter',
               "KJ O'Brien", 'TE Kane', 'EJ Richardson', 'GH Dockrell', 'MC Sorensen', 'Yaqoob Ali']
    match_type = 'ODI'
    inference_row = make_inference_row(team1_names, team2_names, match_type)

    inference_row_copy = inference_row.copy()
    model_input = convert_row(inference_row)
    model_input.unsqueeze(0)
    model_input = (model_input - running_mean) / running_std

    output = model.generate(model_input.unsqueeze(0).to("mps"), 11)
    names = []
    for idx in output:
        names.append(inference_row_copy[8 + idx])
    print(names)