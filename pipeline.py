import torch
from torch.utils.data import DataLoader
from config import load_bpe_tokenizers, Config
from data_utils import TranslationDataset, collate_fn

# initializes tokenizers, datasets, and dataloaders.
# returns the dataloaders and tokenizers for inference later.
def get_dataloaders(train_fr, train_en, val_fr, val_en, batch_size=32):
    # Load the shared tokenizers
    fr_tokenizer, en_tokenizer = load_bpe_tokenizers()
    
    # initialize datasets
    train_dataset = TranslationDataset(train_fr, train_en, fr_tokenizer, en_tokenizer)
    val_dataset = TranslationDataset(val_fr, val_en, fr_tokenizer, en_tokenizer)
    
    # create DataLoaders 
    # shuffle training data, but keep validation sequential
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, fr_tokenizer, en_tokenizer

# dummy test
if __name__ == "__main__":
    dummy_fr = ["Bonjour le monde", "Comment ça va?"]
    dummy_en = ["Hello world", "How are you?"]
    
    # The assignment specifies a batch size of 32 for both models
    train_loader, val_loader, fr_tok, en_tok = get_dataloaders(
        train_fr=dummy_fr, train_en=dummy_en, 
        val_fr=dummy_fr, val_en=dummy_en, 
        batch_size=32 
    )
    
    # get one batch to verify the pipeline works
    for batch in train_loader:
        print("Pipeline Test Successful!")
        print("Source Tensor Shape:", batch["src"].shape)
        print("Target Tensor Shape:", batch["tgt"].shape)
        print("Source Padding Mask Shape:", batch["src_padding_mask"].shape)
        print("Causal Mask Shape:", batch["tgt_causal_mask"].shape)
        break # only test one batch