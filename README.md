# README

To tackle the Hangman word prediction task, a **Transformer Model** setup was deployed and trained from scratch


##  Dataset

Using a custom Python script, all possible hangman states for each word were generated. This yielded approximately **1 billion (xi, yi) pairs**, where:

* **xi**: Masked hangman state of the word
* **yi**: True word

**Final dataset size: \~3 GB**

---

##  Data Engineering

* Each `(xi, yi)` pair is converted into a **PyTorch tensor** of shape `(seq_len, 27)`
* Characters are **one-hot encoded**:

  * `'a'` → index 0
  * `'_'` (masked character) → index 26
* Data is loaded using a **PyTorch DataLoader** with:

  * `batch_size = 512`
  * Padding per batch to match sequence length using zero vectors

---

##  Model Architecture

A standard **Transformer Encoder-Decoder** was used with the following parameters:

* `input_size = 27`
* `d_model = 256`
* `nhead = 8`
* `num_layers = 4`
* **Sinusoidal Positional Encoding**

###  Training Details

* **cross-entropy loss** for backpropagation
* **Optimizer**: Adam (`lr = 1e-3`)
* **Total parameters**: \~2M (\~10 MB)
* **Epochs**: 1 (due to computational constraints)

**Only training for 1 epoch limited model accuracy**


###  Inference

* The model outputs probability distribution over all letters at each index of the word
* The probability distribution of all the masked positions were summed over and the letter 
with max probability and not guessed before is declared as the new guess.


---

##  Results & Conclusion

* Final Accuracy on practice runs (100 runs): **0.62**
* Final Accuracy on recorded runs (1000 runs): **0.56** 
* Baseline model accuracy: **0.18**
* Challenge cutoff: **0.50** 

The model surpassed the baseline and cutoff even after **just 1 epoch** of training. Since transformers typically benefit from longer training or pretrained weights, performance is expected to improve with extended training.

**Many successful transformer-based models are fine-tuned from large pretrained models.** Since the challenge does not allow training on words that are not part of the original corpus, fine-tuning would mean defying this rule. Hence, finetuning option is entirely discarded.

---

##  Python Files

Attached at end of the notebook

| File                   | Description                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| `create_data.py`       | Generates all possible hangman states from `words_250000_train.txt`                          |
| `load_data.py`         | Builds a PyTorch DataLoader from the generated text file                                     |
| `transformer_model.py` | Trains the transformer model and saves it to `.pth` format                                   |
| `predict.py`           | Takes a masked hangman word as input and returns character guesses in descending probability |

---

##  Declaration

All logic and code were written by me. I used **LLMs for debugging** and referencing library syntax when needed.




