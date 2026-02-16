import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Any

class BilingualDataset(Dataset):
    def __init__(self, ds, src_lang, tgt_lang, tokenizer_src, tokenizer_tgt, seq_len):
        super().__init__()

        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding = self.seq_len - len(dec_input_tokens) -1

        enc_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding, dtype=torch.int64),
            ]
        )

        dec_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64),
            ]
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding, dtype=torch.int64),
            ]
        )

        assert enc_input.shape[0] == self.seq_len
        assert dec_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return {
            "enc_input": enc_input, # seq_len
            "dec_input": dec_input, # seq_len
            "enc_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # 1,1,seq_len
            "dec_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.size(0)), # 1,1,seq_len
            "label": label, # seq_len
            "src_text": src_text,  # Original source text
            "tgt_text": tgt_text,  # Original target text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((size, size), dtype=torch.bool), diagonal=1).type(torch.int)
    return mask == 0
    