import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load_data import return_dataloader
from load_data import WordCompletionDataset


class BiLSTMModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=128, num_layers=1):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,        # 26 features per time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,            # (seq_len, batch, input_size)
            bidirectional=True
        )
        
        # Output layer: predicting vector per sequence
        self.fc = nn.Linear(hidden_size * 2, 26)

    def forward(self, packed_input):
        # packed_input: PackedSequence with feature dimension = 26
        packed_input = packed_input.permute(1,0) # (variable , 26)
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # h_n shape: (num_layers * num_directions, batch=1, hidden_size)
        # Remove batch dim with squeeze(1)
        h_forward = h_n[-2, :]  # (hidden_size,)
        h_backward = h_n[-1, :] # (hidden_size,)

        h_concat = torch.cat((h_forward, h_backward), dim=0)  # (hidden_size*2,)

        out = self.fc(h_concat.unsqueeze(0))  
        
        return out.squeeze(1)  

def train(model, dataloader, optimizer, criterion, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, outputs in dataloader:
            # inputs: list of 26 PackedSequences (one per feature)
            # outputs: (batch_size, 26)

            # Move outputs to device
            outputs = outputs.to(device)

            optimizer.zero_grad()

            # Forward pass for each feature separately
            predictions = []
            for i in range(len(inputs)):  # assuming 26 features
                packed_input = torch.tensor(inputs[i], dtype = torch.float32).to(device)
                pred = model(packed_input)  # model should return (batch_size,) or (batch_size, 1)
                predictions.append(pred)

            # Stack predictions to match output shape: (batch_size, 26)
            predictions = torch.stack(predictions, dim=1)
            predictions = predictions.squeeze(0)
            # Compute loss
            loss = criterion(predictions.float(), outputs.float())

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "trained_model.pth")
    

if __name__ == "__main__":


    dataset, dataloader = return_dataloader()
    model = BiLSTMModel()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  

    train(model, dataloader, optimizer, criterion, 3, 'cuda')
