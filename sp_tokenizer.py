## max length 175
# type: ignore[import]
import sentencepiece as spm
import torch
from typing import List

seq_length = 175

sp = spm.SentencePieceProcessor(model_file='tokenizer_model.model')
assert sp.pad_id() == 0

def encode(text: str) -> List[int]:
    encoded = sp.encode_as_ids(text)
    if len(encoded) > seq_length - 1:
        encoded = encoded[:seq_length-1]
    return encoded

def encode_tgt(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = encode(text)

    tgt_in = list(encoded)
    tgt_in.insert(0, sp.bos_id())
    
    tgt_out = list(encoded)
    tgt_out.append(sp.eos_id())

    return (torch.tensor(tgt_in, dtype=torch.long), torch.tensor(tgt_out, dtype=torch.long))

def decode(ids_tensor: torch.Tensor) -> str:  
    ids = ids_tensor.tolist()
    if len(ids) == 0:
        return ""
    if ids[0] == sp.bos_id():
        ids = ids[1:]

    for i, token_id in enumerate(ids):
        if token_id == sp.eos_id():
            ids = ids[:i]
            break
        
            
    ids = [token_id for token_id in ids if token_id != sp.pad_id()]

    decoded = sp.decode_ids(ids)

    return decoded


print(decode(torch.tensor(encode("Hello"))))
