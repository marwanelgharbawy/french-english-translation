import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

class TranslationDataset(Dataset):
    def __init__(self, fr_sentences, en_sentences, fr_tokenizer, en_tokenizer):
        self.fr_sentences = fr_sentences
        self.en_sentences = en_sentences
        self.fr_tokenizer = fr_tokenizer
        self.en_tokenizer = en_tokenizer

    def __len__(self):
        return len(self.fr_sentences)

    def __getitem__(self, idx):
        # tokenize strings
        fr_tokens = self.fr_tokenizer.encode(self.fr_sentences[idx])
        en_tokens = self.en_tokenizer.encode(self.en_sentences[idx])
        
        # add BOS at start and EOS at end
        fr_tokens = [Config.BOS_TOKEN_ID] + fr_tokens + [Config.EOS_TOKEN_ID]
        en_tokens = [Config.BOS_TOKEN_ID] + en_tokens + [Config.EOS_TOKEN_ID]
        
        return torch.tensor(fr_tokens, dtype=torch.long), torch.tensor(en_tokens, dtype=torch.long)

def generate_causal_mask(seq_len):
    # creates a lower triangular matrix: 
    # - True for past and current tokens
    # - False for future tokens
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask

# it's called internally by the DataLoader to collate a batch of data points together
# collate used to take a batch of variable length sequences together, pad them, and create masks
# returns a dictionary with necessary tensors for the model
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    
    # truncate to MAX_SEQ_LEN
    for src_item, tgt_item in batch:
        src_item = src_item[:Config.MAX_SEQ_LEN]
        tgt_item = tgt_item[:Config.MAX_SEQ_LEN]
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
        
    # pad sequences to the same length in the batch
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=Config.PAD_TOKEN_ID)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=Config.PAD_TOKEN_ID)
    
    # create padding masks
    # True means it is a real token, False means it is a pad token
    src_padding_mask = (src_padded != Config.PAD_TOKEN_ID)
    tgt_padding_mask = (tgt_padded != Config.PAD_TOKEN_ID)
    
    # create causal mask for the target sequence (Max_Seq_Len, Max_Seq_Len)
    tgt_causal_mask = generate_causal_mask(tgt_padded.size(1))
    
    # return a dictionary with necessary tensors for the model
    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask,
        "tgt_causal_mask": tgt_causal_mask
    }