import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import return_dataloader
from load_data import WordCompletionDataset


class TransformerModel(nn.Module):
    def __init__(self, input_size=26, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=False  # (seq_len, batch, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # FCN head
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, 26)

    def forward(self, input):
        # input shape: (26, variable) → interpret as (feature, seq_len)
        input = input.permute(1, 0)  # shape: (seq_len, feature)
        input = input.unsqueeze(1)  # (seq_len, batch=1, feature)
        
        x = self.embedding(input)   # (seq_len, batch=1, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch=1, d_model)

        # Aggregate sequence — e.g., mean over sequence length
        x = x.mean(dim=0)  # (batch=1, d_model)

        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)

        out = self.fc2(x)
        out = F.softmax(out, dim=1)

        return out  # shape: (1, 26)


def train(model, dataloader, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(dataloader)
        batch = 0
        for inputs, outputs in dataloader:
            # inputs: list of 26 PackedSequences (one per feature)
            # outputs: (batch_size, 26)

            # Move outputs to device
            outputs = outputs.to(device)

            optimizer.zero_grad()

            # Forward pass for each feature separately
            predictions = []
            for i in range(len(inputs)):  # assuming 26 features
                packed_input = inputs[i].detach().clone().float().to(device)
                pred = model(packed_input)  # model should return (batch_size,) or (batch_size, 1)
                predictions.append(pred)

            # Stack predictions to match output shape: (batch_size, 26)
            predictions = torch.stack(predictions, dim=1)
            predictions = predictions.squeeze(0)
            # Compute loss
            loss = -torch.sum(outputs.float() * torch.log(predictions.float()+ 1e-8))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch += 1
            if (batch % 100 == 0) :
                print(f'Epoch: {epoch+1} - {batch} batch done of total {total_batches} batches...({batch/total_batches * 100 :.2f}%)')

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "trained_model.pth")
    

if __name__ == "__main__":


    dataset, dataloader = return_dataloader()
    model = TransformerModel()

    print("Number of trainable paramaters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, dataloader, optimizer, 1, 'cuda')
