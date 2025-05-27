import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pack_sequence

class WordCompletionDataset(Dataset):
    def __init__(self, filepath):
        self.samples = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                # if (i % 1000 == 0) :
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

    

    packed_inputs = [
        i if isinstance(i, torch.Tensor) else torch.tensor(i, dtype=torch.float32)
        for i in inputs
    ]

    outputs_tensor = torch.stack([
        o if isinstance(o, torch.Tensor) else torch.tensor(o, dtype=torch.float32)
        for o in outputs
    ])

    return packed_inputs, outputs_tensor



def return_dataloader():
    dataset = WordCompletionDataset("small_strip_25000.txt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    print("Dataset Loaded Successfully")
    return dataset, dataloader

if __name__ == "__main__": 
    dataset = WordCompletionDataset("small_strip.txt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    for inputs, outputs in dataloader:
        # inputs: list of 26 PackedSequence objects (one per feature)
        # outputs: shape (batch_size, 26)
        print(len(dataloader))
        print(f"Inputs shape: {len(inputs)}, {inputs[0].shape}")
        print(f"Outputs shape: {outputs.shape}")  # (batch_size, 26)

        break  # Only process first batch

