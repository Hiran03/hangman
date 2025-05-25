import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WordCompletionDataset(Dataset):
    def __init__(self, filepath):
        self.samples = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                input_word, output_word = line.strip().split()
                self.samples.append((input_word, output_word))

        
    def input_encode(self, word):
        matrix = np.zeros((26,len(word)))
        for i in range(len(word)):
            if word[i] != '_':
                matrix[ord(word[i]) - ord('a')][i] = 1
        return matrix
    def input_decode(self, matrix):
        word = []
        for i in range(matrix.shape[1]):  # iterate over columns (positions in the word)
            col = matrix[:, i]
            if np.sum(col) == 0:
                word.append('_')  # no letter guessed at this position
            else:
                letter_index = np.argmax(col)
                word.append(chr(letter_index + ord('a')))
        return ''.join(word)

        
    def output_encode(self, input_word, output_word):
        missing_letters = []
        for i in range(len(input_word)) :
            if input_word[i] == '_' :
                missing_letters.append(output_word[i])
        array = np.zeros(26)
        for i in missing_letters:
            array[ord(i) - ord('a')] += 1
        array /= np.sum(array)
        return array    
    def output_decode(self, array):
        indices = np.where(array > 0)[0]  # get indices where probability > 0
        letters = [chr(i + ord('a')) for i in indices]
        return letters
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_word, output_word = self.samples[idx]
        input_ids = self.input_encode(input_word)
        output_ids = self.output_encode(input_word, output_word)
        return torch.tensor(input_ids), torch.tensor(output_ids)

def collate_fn(batch):
    inputs, outputs = zip(*batch)
    return inputs, outputs

# Assuming your dataset class and collate_fn are already defined, and your class instance is `self`

def return_dataloader():
    dataset = WordCompletionDataset("small_strip.txt")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print("Dataset Loaded Successfully")
    return dataset, dataloader

# for inputs, outputs in dataloader:
#     print(f"Inputs shape: {len(inputs)}, {len(inputs[0])}, {len(inputs[0][0])}")   # e.g. (64, 26, seq_len)
#     print(f"Outputs shape: {len(outputs)}, {len(outputs[0])}") # e.g. (64, 26)

#     # Decode first batch sample (index 0)
#     first_input_matrix = inputs[1].numpy()  # (26, seq_len)
#     first_output_array = outputs[1].numpy() # (26,)

#     # Transpose input to (26, seq_len) if needed (depends on your encode shape)
#     # For your input_encode, shape was (26, len(word)), so inputs should be (batch, 26, seq_len)
#     decoded_input = dataset.input_decode(first_input_matrix)
#     decoded_output = dataset.output_decode(first_output_array)

#     print(f"Decoded input (sample 0): {decoded_input}")
#     print(f"Decoded output (sample 0): {decoded_output}")

#     break  # Only first batch

