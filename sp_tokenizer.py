

## max length 175
# type: ignore[import]
import sentencepiece as spm
import torch
from typing import List

seq_length = 175

class SentencePieceTokenizer:
    def __init__(self, model_file: str = 'tokenizer_model.model', seq_length: int = seq_length) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        assert self.sp.pad_id() == 0
        self.seq_length = seq_length

    def encode(self, text: str) -> List[int]:
        encoded = self.sp.encode_as_ids(text)
        if len(encoded) > self.seq_length - 1:
            encoded = encoded[:self.seq_length-1]
        return encoded

    def encode_tgt(self, text: str) -> tuple[list[int], list[int]]:
        encoded = self.encode(text)

        tgt_in = list(encoded)
        tgt_in.insert(0, self.sp.bos_id())
        
        tgt_out = list(encoded)
        tgt_out.append(self.sp.eos_id())

        return (tgt_in, tgt_out)

    def decode(self, ids_tensor: torch.Tensor) -> str:  
        ids = ids_tensor.tolist()
        if len(ids) == 0:
            return ""
        if ids[0] == self.sp.bos_id():
            ids = ids[1:]

        for i, token_id in enumerate(ids):
            if token_id == self.sp.eos_id():
                ids = ids[:i]
                break
            
            
        ids = [token_id for token_id in ids if token_id != self.sp.pad_id()]

        decoded = self.sp.decode_ids(ids)

        return decoded


if __name__ == "__main__":
    tokenizer = SentencePieceTokenizer()
    print(tokenizer.decode(torch.tensor(tokenizer.encode("Hello"))))
