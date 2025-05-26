from model import BiLSTMModel
import torch
from load_data import WordCompletionDataset
model = BiLSTMModel()

model.load_state_dict(torch.load("trained_model.pth"))
model.eval()  # if you're doing inference

predictions = model(torch.tensor(WordCompletionDataset("small_strip.txt").input_encode("h_ran"), dtype = torch.float32))

# predictions: torch.Tensor of shape (26, 1)
sorted_indices = torch.argsort(predictions.squeeze(), descending=True).tolist()
guesses = [chr(i + ord('a')) for i in sorted_indices]

print(predictions, sorted_indices)
print(guesses)