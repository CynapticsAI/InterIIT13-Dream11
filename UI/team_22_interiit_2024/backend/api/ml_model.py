import torch
import torch.nn as nn
import torch.utils.data as data
import math
from backend.settings import BASE_DIR
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(os.path.join(BASE_DIR, "api/model_config.json"), "r") as f:
    data = json.load(f)
    running_mean = data["running_mean"]
    running_std = data["running_std"]
    d_model = data["d_model"]
    n_heads = data["n_heads"]
    n_layers = data["n_layers"]

model = GPT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "api/final_model.pt"), map_location=device))
model.eval()

def predict_players(inference_row):
    inference_row_copy = inference_row.copy()
    model_input = convert_row(inference_row)
    model_input = (model_input - running_mean) / running_std

    model_input = model_input.unsqueeze(0).to(device)
    output = model.generate(model_input, 11)

    names = []
    for idx in output:
        names.append(inference_row_copy[8 + idx])
    return names