import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import return_dataloader
from load_data import WordCompletionDataset
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size=27, d_model=64, nhead=2, num_layers=2, dim_feedforward=256, max_len=100):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, input_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(x)
        return output  # (batch, seq_len, 27)


def train(model, dataloader, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(dataloader)
        batch = 0

        for inputs, outputs, key_padding_mask in dataloader:
            inputs, outputs = inputs.to(device), outputs.to(device)

            # Compute mask for padding positions
            src_key_padding_mask = (inputs == 0).all(dim=-1).bool()  # (batch, seq_len)

            optimizer.zero_grad()
            predictions = model(inputs, src_key_padding_mask=src_key_padding_mask)

            # Apply softmax over last dimension
            predictions = F.log_softmax(predictions, dim=-1)

            # Focus loss only on masked positions
            mask_token_index = 26
            masked_positions = (inputs[:, :, mask_token_index] == 1).unsqueeze(-1)  # (batch, seq_len, 1)

            loss = -torch.sum(masked_positions * outputs * predictions)
            loss = loss / masked_positions.sum().clamp(min=1)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch += 1
            if (batch % 100 == 0) :
                print(f'Epoch: {epoch+1} - {batch} batch done of total {total_batches} batches...({batch/total_batches * 100 :.2f}%)')
            if (batch % 1000 == 0):
                torch.save(model.state_dict(), "trained_model.pth")
                print("Model saved")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "trained_model.pth")


if __name__ == "__main__":
    dataset, dataloader = return_dataloader()
    model = TransformerModel()
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, dataloader, optimizer, 1, device)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from load_data import return_dataloader
# import math
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=100):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         seq_len = x.size(0)
#         return x + self.pe[:seq_len]

# class TransformerModel(nn.Module):
#     def __init__(self, input_size=27, d_model=64, nhead=2, num_layers=2, dim_feedforward=256, max_len=100):
#         super(TransformerModel, self).__init__()

#         self.embedding = nn.Linear(input_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
#         self.attn_window = 3

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=0.1,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.decoder = nn.Linear(d_model, input_size)

#     def forward(self, x, key_padding_mask=None):
#         x = self.embedding(x)              # (seq_len, batch, d_model)
#         x = self.pos_encoder(x)            # add positional encoding

#         # Local attention mask
#         seq_len = x.size(0)
#         mask = torch.full((seq_len, seq_len), float('-inf')).to(x.device)
#         for i in range(seq_len):
#             for j in range(max(0, i - self.attn_window), min(seq_len, i + self.attn_window + 1)):
#                 mask[i, j] = 0

#         x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=key_padding_mask)
#         x = self.decoder(x)
#         x = F.softmax(x, dim=-1)
#         return x  # (seq_len, batch, 27)



# def train(model, dataloader, optimizer, num_epochs, device):
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         total_batches = len(dataloader)
#         batch = 0
#         for inputs, outputs, key_padding_mask in dataloader:
#             inputs = inputs.to(device)  # (batch, seq_len, 27)
#             outputs = outputs.to(device)
#             key_padding_mask = key_padding_mask.to(device)

#             inputs = inputs.permute(1, 0, 2)  # (seq_len, batch, 27)
#             outputs = outputs.permute(1, 0, 2)  # (seq_len, batch, 27)

#             optimizer.zero_grad()
#             predictions = model(inputs, key_padding_mask=~key_padding_mask)  # (seq_len, batch, 27)

#             # Focus loss on masked positions only
#             mask = inputs[:, :, 26].unsqueeze(-1)  # (seq_len, batch, 1)
#             masked_outputs = outputs * mask
#             masked_predictions = predictions * mask
#             loss = -torch.sum(masked_outputs * torch.log(masked_predictions + 1e-8)) / torch.sum(mask)

#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             batch += 1
#             if (batch % 100 == 0) :
#                 print(f'Epoch: {epoch+1} - {batch} batch done of total {total_batches} batches...({batch/total_batches * 100 :.2f}%)')
#             if (batch % 1000 == 0):
#                 torch.save(model.state_dict(), "trained_model.pth")
#                 print("Model saved")
                
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
#     torch.save(model.state_dict(), "trained_model.pth")

# if __name__ == "__main__":
#     dataset, dataloader = return_dataloader()
#     model = TransformerModel()
#     print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     train(model, dataloader, optimizer, num_epochs=1, device=device)