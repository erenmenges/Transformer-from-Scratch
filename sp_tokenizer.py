## max length 175
# type: ignore[import]
import sentencepiece as spm
import torch
import copy
from typing import List

seq_length = 175

sp = spm.SentencePieceProcessor(model_file='tokenizer_model.model')
print(sp.pad_id(), sp.bos_id(), sp.eos_id())

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

    for _ in range(seq_length - len(tgt_in)):
        tgt_in.append(sp.pad_id())

    for _ in range(seq_length - len(tgt_out)):
        tgt_out.append(sp.pad_id())

    return (torch.tensor(tgt_in), torch.tensor(tgt_out))

def decode(ids: torch.Tensor) -> str:

    for i, token_id in enumerate(ids):
        if token_id.item() == sp.eos_id():
            if ids[0] == sp.bos_id():
                ids = ids[1:i]
            else:
                ids = ids[:i]
            break

    decoded = sp.decode_ids(ids.tolist())

    return decoded


print(decode(torch.tensor(encode("Hello"))))
