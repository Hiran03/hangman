import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

from load_data import return_dataloader

dataset, dataloader = return_dataloader()

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=128, num_layers=1, output_size=26):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,   # input: (seq_len, batch, input_size)
                            bidirectional=True)
        
        # Fully connected layer to output probabilities over letters
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
    
    def forward(self, x, lengths):
        """
        x: Tensor of shape (batch_size, input_size=26, seq_len)
        lengths: List/Tensor of sequence lengths (for packing)
        """
        # Rearrange input to (seq_len, batch_size, input_size)
        x = x.permute(2, 0, 1)  # (seq_len, batch, 26)

        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # For bidirectional, directions=2
        # We want to concatenate forward and backward hidden states from last layer
        # So take last layer's forward and backward states:
        h_forward = h_n[-2,:,:]  # (batch, hidden_size)
        h_backward = h_n[-1,:,:] # (batch, hidden_size)

        h_concat = torch.cat((h_forward, h_backward), dim=1)  # (batch, hidden_size*2)

        # Fully connected layer
        output = self.fc(h_concat)  # (batch, output_size=26)

        return output

model = BiLSTMModel()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()  # expects targets as class indices

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, outputs in dataloader:
        inputs = inputs.to(device)          # (batch, 26, seq_len)
        outputs = outputs.to(device)        # (batch, 26)


        optimizer.zero_grad()

        preds = model(inputs)   # (batch, 26)

        # Convert output probabilities to class indices for CrossEntropyLoss
        # If outputs are one-hot or soft labels, do argmax along dim=1:
        target_labels = torch.argmax(outputs, dim=1)  # (batch,)

        loss = criterion(preds, target_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

