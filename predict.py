from transformer_model import TransformerModel
import torch
from load_data import WordCompletionDataset

import torch
import torch.nn.functional as F
from transformer_model import TransformerModel
from load_data import WordCompletionDataset
import numpy as np

def make_predictions(word, max_len=64):
    dataset = WordCompletionDataset("small_strip_25000.txt")
    encoded = dataset.input_encode(word)  # (27, seq_len)
    seq_len = encoded.shape[1]

    # Pad to max_len
    if seq_len < max_len:
        pad_width = ((0, 0), (0, max_len - seq_len))
        encoded = np.pad(encoded, pad_width, mode='constant', constant_values=0)

    input_tensor = torch.tensor(encoded, dtype=torch.float32)  # (27, max_len)
    input_tensor = input_tensor.permute(1, 0).unsqueeze(1)     # (max_len, 1, 27)

    # Padding mask: True where position is padded (all zeros)
    pad_mask = (encoded.sum(axis=0) == 0)                      # (max_len,)
    pad_mask_tensor = torch.tensor(pad_mask).unsqueeze(0)     # (1, max_len)

    model = TransformerModel()
    model.load_state_dict(torch.load("trained_model.pth", map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor, key_padding_mask=pad_mask_tensor)  # (max_len, 1, 27)
        output = output.squeeze(1).T  # (27, max_len)

    # Extract masked positions
    mask_indices = np.where(encoded[26] == 1)[0]
    if len(mask_indices) == 0:
        print("No masked characters found in input.")
        return []

    masked_preds = output[:, mask_indices].mean(dim=1)  # (27,)
    masked_preds[26] = -float('inf')  # Mask token shouldn't be predicted

    sorted_indices = torch.argsort(masked_preds, descending=True).tolist()
    guesses = [chr(i + ord('a')) for i in sorted_indices]

    return guesses



if __name__ == '__main__':
    print(make_predictions('hi_an'))