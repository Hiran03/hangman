from transformer_model import TransformerModel
import torch
from load_data import WordCompletionDataset

import torch
import torch.nn.functional as F
from transformer_model import TransformerModel
from load_data import WordCompletionDataset
import numpy as np

import torch
import numpy as np
from transformer_model import TransformerModel
from load_data import WordCompletionDataset

def make_predictions(word, max_len=64):
    dataset = WordCompletionDataset("small_strip_25000.txt")
    encoded = dataset.input_encode(word)  # (27, seq_len)
    seq_len = encoded.shape[1]

    # Pad to max_len
    if seq_len < max_len:
        pad_width = ((0, 0), (0, max_len - seq_len))
        encoded = np.pad(encoded, pad_width, mode='constant', constant_values=0)

    input_tensor = torch.tensor(encoded, dtype=torch.float32).T.unsqueeze(0)  # (1, max_len, 27)

    # Create src_key_padding_mask: True where position is padding
    pad_mask = (input_tensor.sum(-1) == 0)  # (1, max_len)

    model = TransformerModel()
    model.load_state_dict(torch.load("trained_model.pth", map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor, src_key_padding_mask=pad_mask)  # (1, max_len, 27)
        output = output.squeeze(0).T  # (27, max_len)

    # Extract masked positions (channel 26 indicates masked token)
    mask_indices = np.where(encoded[26] == 1)[0]
    if len(mask_indices) == 0:
        print("No masked characters found in input.")
        return []

    # Average logits across masked positions
    masked_preds = output[:, mask_indices].mean(dim=1)  # (27,)
    masked_preds[26] = -float('inf')  # prevent predicting [MASK]

    sorted_indices = torch.argsort(masked_preds, descending=True).tolist()
    guesses = [chr(i + ord('a')) for i in sorted_indices]

    return guesses



if __name__ == '__main__':
    print(make_predictions('hi_an'))