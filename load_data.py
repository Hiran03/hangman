import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class WordCompletionDataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.line_offsets = []

        # Precompute line start offsets for fast random access
        with open(filepath, 'rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)

    def input_encode(self, word):
        matrix = np.zeros((27, len(word)))
        for i, ch in enumerate(word):
            if ch != '_':
                matrix[ord(ch) - ord('a')][i] = 1
            else:
                matrix[26][i] = 1
        return matrix

    def output_encode(self, word):
        matrix = np.zeros((27, len(word)))
        for i, ch in enumerate(word):
            matrix[ord(ch) - ord('a')][i] = 1
        return matrix

    def input_decode(self, matrix):
        word = []
        matrix = np.array(matrix)
        for i in range(matrix.shape[1]):
            idx = np.argmax(matrix[:, i])
            word.append('_' if idx == 26 else chr(idx + ord('a')))
        return ''.join(word)

    def output_decode(self, matrix):
        word = []
        matrix = np.array(matrix)
        for i in range(matrix.shape[1]):
            idx = np.argmax(matrix[:, i])
            word.append(chr(idx + ord('a')))
        return ''.join(word)

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        offset = self.line_offsets[idx]
        with open(self.filepath, 'r') as f:
            f.seek(offset)
            line = f.readline().strip()
            input_word, output_word = line.split()

        input_ids = self.input_encode(input_word)
        output_ids = self.output_encode(output_word)
        return torch.tensor(input_ids, dtype=torch.float32), torch.tensor(output_ids, dtype=torch.float32)

def collate_fn(batch):
    inputs, outputs = zip(*batch)
    inputs = [i.permute(1, 0) for i in inputs]
    outputs = [o.permute(1, 0) for o in outputs]

    padded_inputs = pad_sequence(inputs, batch_first=True)  # (batch, seq_len, 27)
    padded_outputs = pad_sequence(outputs, batch_first=True)  # (batch, seq_len, 27)

    key_padding_mask = (padded_inputs.sum(-1) != 0)  # (batch, seq_len)
    return padded_inputs, padded_outputs, key_padding_mask

def return_dataloader():
    dataset = WordCompletionDataset("small_strip_250000.txt")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=collate_fn)
    print("Dataset Loaded Successfully")
    return dataset, dataloader


if __name__ == "__main__": 
    dataset = WordCompletionDataset("small_strip_250000.txt")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

    for inputs, outputs, _ in dataloader:
        print(f"Number of batches: {len(dataloader)}")
        print(f"Inputs batch shape: {inputs.shape}")   # (batch_size, seq_len, 27)
        print(f"Outputs batch shape: {outputs.shape}") # (batch_size, seq_len, 27)

        # Visualize one word pair
        idx = 0  # first sample in batch
        input_matrix = inputs[idx].permute(1, 0).numpy()   # (27, seq_len)
        output_matrix = outputs[idx].permute(1, 0).numpy() # (27, seq_len)

        input_word = dataset.input_decode(input_matrix)
        output_word = dataset.output_decode(output_matrix)

        print(f"\nSample {idx + 1}")
        print(f"Masked Input Word:  {input_word}")
        print(f"Ground Truth Word:  {output_word}")
        
        break