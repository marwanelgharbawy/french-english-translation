# French to English Translation - RNN and Transformers

### Required

PyTorch and Transformers libraries are required.

```bash
pip install torch transformers
```

### Data Setup

Three directories are required in the current working directory: `tokenizer_en`, `tokenizer_fr`, and `parallel_en_fr_corpus`. They are downloaded from this [link](https://drive.google.com/drive/folders/1F6JLDU6EBGWgxps4sP7b4JJIS6OosVTW).

### File Structure

#### `config.py`

Centralizes the shared configuration. It contains the `Config` dataclass for special token IDs (PAD, BOS, EOS, UNK) and the sequence length, as well as the function to load the BPE tokenizers.

#### `data_utils.py`

Contains the core PyTorch logic. It holds the `TranslationDataset` class for tokenizing and injecting special tokens, and the `collate_fn` for padding batches and generating attention masks.

#### `pipeline.py`
Contains the `get_dataloaders` wrapper function that initializes the tokenizers and returns the fully configured train and validation DataLoaders.

### Usage

Use the `get_dataloaders` function to initialize the pipeline. It returns the DataLoaders and the tokenizers.

```python
from pipeline import get_dataloaders

# Read the text files into lists of strings
with open("parallel_en_fr_corpus/train.fr", "r", encoding="utf-8") as f:
    train_fr_data = f.read().splitlines()
with open("parallel_en_fr_corpus/train.en", "r", encoding="utf-8") as f:
    train_en_data = f.read().splitlines()
    
# Repeat the above for validation data (val.fr, val.en)

# Initialize the pipeline (batch size defaults to 32)
train_loader, val_loader, fr_tokenizer, en_tokenizer = get_dataloaders(
    train_fr=train_fr_data, 
    train_en=train_en_data,
    val_fr=val_fr_data,
    val_en=val_en_data
)

# Iterate through batches during training
for batch in train_loader:
    src = batch["src"]                               # (batch_size, max_seq_len)
    tgt = batch["tgt"]                               # (batch_size, max_seq_len)
    src_padding_mask = batch["src_padding_mask"]     # (batch_size, max_seq_len)
    tgt_padding_mask = batch["tgt_padding_mask"]     # (batch_size, max_seq_len)
    tgt_causal_mask = batch["tgt_causal_mask"]       # (max_seq_len, max_seq_len)
    
    # Pass to model.encode() / model.decode_step()
```