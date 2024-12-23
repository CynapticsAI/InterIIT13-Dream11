import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
import torch


class D11Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, row):
        # Get match data and expand it for all 22 players
        match_data = torch.tensor(self.data.iloc[row, 1:8], dtype=torch.float32).unsqueeze(0).expand(22, 7)

        # Process target data for each player (22 players, 15 stats)
        player_targets = []
        player_idxs = self.data.iloc[row, 30:360][14::15].argsort()[::-1]

        for i in player_idxs:
            temp = torch.zeros(192)
            temp[100:115] = torch.tensor(self.data.iloc[row, 30+i*15:45+i*15], dtype=torch.float32) 
            temp[115+i] = 1
            player_targets.append(temp)
        target = torch.stack(player_targets)  # Shape: [22, 192]

        # Process source data for each player (22 players, 33 stats)
        head_to_head = []
        for i in range(22):
            head_to_head_stats = self.data.iloc[row, 360 + (i*33): 393 + (i*33)]
            head_to_head.append(torch.tensor(head_to_head_stats, dtype=torch.float32))

        head_to_head = torch.stack(head_to_head)  # Shape: [22, 33]

        player_data = []
        for i in range(22):
            player_stats = self.data.iloc[row, 1086 + (i*60):1146 + (i*60)]
            player_data.append(torch.tensor(player_stats, dtype=torch.float32))
        player_data = torch.stack(player_data)  # Shape: [22, 60]

        # Concatenate match data with player data
        source = torch.cat([match_data, player_data, head_to_head], dim=1)  # Shape: [22, 100]
        src = torch.zeros((22, 192))
        src[:, :100] = source
        tokens = torch.cat([src, target]) # Shape: [44, 100]

        return {"src":tokens[:-1], "tgt":tokens[1:]}

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

if __name__ == "__main__":
    d_model = 192 # the dimension of the embeddings
    n_heads = 4 # number of self-attention heads. should be divisible with d_model
    n_layers = 4 # number of gpt blocks/layers
    m = GPT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to("mps")
    m.train()
    print(m)
    print(f"Total Parameters: {round(sum(p.numel() for p in m.parameters() if p.requires_grad))}")

    df = pd.read_csv("../data/processed/match_data.csv")
    train_df = df[pd.to_datetime(df["date"]) <= pd.to_datetime("2024-06-30")]
    test_df = df[pd.to_datetime(df["date"]) > pd.to_datetime("2024-06-30")]

    lr = 1e-2
    optim = torch.optim.AdamW(m.parameters(), lr=lr)
    train_dataset = D11Dataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_dataset = D11Dataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    unnorm_dataset = D11Dataset(test_df)
    unnorm_dataloader = DataLoader(unnorm_dataset, batch_size=1, shuffle=False)

    import time
    import tqdm

    epochs = 1
    eval_steps = 200 # perform evaluation in every n steps
    losses = []
    pbar = tqdm.tqdm(total=len(train_dataloader)*epochs)
    running_mean = 0
    running_std = 1
    for ep in range(epochs):
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()

            xb, yb = batch["src"], batch["tgt"]
            running_mean = xb.mean() * 0.01 + running_mean * 0.99
            running_std = xb.std() * 0.01 + running_std * 0.99
            # normalize
            xb = ((xb - running_mean) / running_std).to("mps")
            yb = ((yb - running_mean) / running_std).to("mps")

            logits, loss = m(xb, yb)
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.4f}")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(loss.item())

            if step == 0 or step % eval_steps == 0:
                m.eval()
                mae = 0
                print("Validating...")
                val_bar = tqdm.tqdm(total=len(test_dataloader))
                for i, (input_batch, output_batch) in enumerate(zip(test_dataloader, unnorm_dataloader)):
                    xvb, yvb = input_batch["src"], input_batch["tgt"]
                    # normalize
                    
                    xvb = ((xvb - running_mean) / running_std).to("mps")
                    yvb = ((yvb - running_mean) / running_std).to("mps")

                    output_batch = output_batch["tgt"].to("mps")
                    predicted_indices = m.generate(xvb[0:, :22, :], 11)
                    actual_score = sum([id.item() for id in output_batch[0, 21:, 114].sort()[0]][:10:-1])
                    predicted_score = output_batch[0, 21:, 114][predicted_indices].sum()
                    assert predicted_score <= actual_score
                    mae += abs(actual_score - predicted_score)
                    if i == 0:
                        print(f"Predicted: {predicted_score}\tActual: {actual_score}")
                    val_bar.update(1)
                    val_bar.set_description(f"MAE: {mae / (i+1):.4f}")
                    if i == 25:
                        val_bar.close()
                        break
                mae = (mae / (i+1))

                end_time = time.time()  # Record the end time
                elapsed_time = end_time - start_time  # Calculate the elapsed time

                print(f"Epoch: {ep}\tlr: {lr}\ttrain_loss: {loss:.4f}\tMAE: {mae:.4f}\ttime: {elapsed_time:.2f} sec")

                m.train()  # Back to training mode

    pbar.close()
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    # Save the model
    torch.save(m.state_dict(), "../model_artifacts/final_model.pt")

    # Save the model config (D_model, n_heads, n_layers, running_mean, running_std)
    import json 
    data = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "running_mean": running_mean.item(),
        "running_std": running_std.item()
    }
    with open("../model_artifacts/model_config.json", "w") as f:
        json.dump(data, f)


    # Load the model
    m.load_state_dict(torch.load("../model_artifacts/final_model.pt"))

    # Load the running mean and std
    with open("../model_artifacts/model_config.json", "r") as f:
        data = json.load(f)
        running_mean = data["running_mean"]
        running_std = data["running_std"]
        d_model = data["d_model"]
        n_heads = data["n_heads"]
        n_layers = data["n_layers"]

    m = GPT(d_model=d_model, n_heads=n_heads, n_layers=n_layers).to("mps")
    # Generate predictions
    m.eval()
    test_dataset = D11Dataset(test_df)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    unnorm_dataset = D11Dataset(test_df)
    unnorm_dataloader = DataLoader(unnorm_dataset, batch_size=1, shuffle=False)
    mae = 0
    for i, (input_batch, output_batch) in enumerate(zip(test_dataloader, unnorm_dataloader)):
        xvb, yvb = input_batch["src"], input_batch["tgt"]
        # normalize
        xvb = ((xvb - running_mean) / running_std).to("mps")
        yvb = ((yvb - running_mean) / running_std).to("mps")

        output_batch = output_batch["tgt"].to("mps")
        predicted_indices = m.generate(xvb[0:, :22, :], 11)
        actual_score = sum([id.item() for id in output_batch[0, 21:, 114].sort()[0]][:10:-1])
        predicted_score = output_batch[0, 21:, 114][predicted_indices].sum()
        assert predicted_score <= actual_score
        mae += abs(actual_score - predicted_score)
        if i == 0:
            print(f"Predicted: {predicted_score}\tActual: {actual_score}")

    mae = (mae / (i+1))
    print(f"MAE: {mae:.4f}")
