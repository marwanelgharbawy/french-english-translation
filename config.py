from dataclasses import dataclass
from unicodedata import name
from transformers import PreTrainedTokenizerFast

# better than enum: access it using Config.PAD_TOKEN_ID for example
@dataclass
class Config:
    PAD_TOKEN_ID: int = 0 # padding
    BOS_TOKEN_ID: int = 1 # beginning of Sequence
    EOS_TOKEN_ID: int = 2 # end of Sequence
    UNK_TOKEN_ID: int = 3 # unknown
    
    MAX_SEQ_LEN: int = 32

def load_bpe_tokenizers(fr_tokenizer_path="tokenizer_fr/tokenizer.json", en_tokenizer_path="tokenizer_en/tokenizer.json"):
    
    # load BPE tokenizers, string to int mapping
    fr_tokenizer = PreTrainedTokenizerFast(tokenizer_file=fr_tokenizer_path)
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file=en_tokenizer_path)
    
    # use the correct padding token id for both tokenizers
    fr_tokenizer.pad_token_id = Config.PAD_TOKEN_ID
    en_tokenizer.pad_token_id = Config.PAD_TOKEN_ID
    
    return fr_tokenizer, en_tokenizer

if __name__ == "__main__":
    fr_tokenizer, en_tokenizer = load_bpe_tokenizers()
    print("French tokenizer vocab size:", len(fr_tokenizer))
    print("English tokenizer vocab size:", len(en_tokenizer))