import torch
from sp_tokenizer import SentencePieceTokenizer


class dataset30k:
    def __init__(self, src_path: str, tgt_path:str):
        with open(src_path, 'r', encoding='utf-8') as f_src:
            src_lines = [line.strip() for line in f_src]
        with open(tgt_path, 'r', encoding='utf-8') as f_tgt:
            tgt_lines = [line.strip() for line in f_tgt]

        assert len(src_lines) == len(tgt_lines), "Source and target files must have the same number of lines"

        self.src_texts = src_lines
        self.tgt_texts = tgt_lines
        self.tokenizer = SentencePieceTokenizer(model_file='tokenizer_model.model')
        self.size = len(src_lines)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return (self.src_texts[idx], self.tgt_texts[idx])

    def maxlen(self):
        maxLength = 0
        concattedList = self.src_texts + self.tgt_texts
        for seq in concattedList:
            if len(seq) > maxLength:
                maxLength = len(seq)
        return maxLength
    
    @staticmethod
    def pad_right(seqs: list[list[int]], pad_id: int):
        maxLengthInBatch = max(len(s) for s in seqs)
        for s in seqs:
            for i in range(len(s), maxLengthInBatch):
                s.append(pad_id)
        return seqs

    @staticmethod
    def make_padding_mask(padded: torch.LongTensor, pad_id: int) -> torch.Tensor:
        return padded == pad_id

    @staticmethod
    def make_triangle_mask(tgt_len: int) -> torch.Tensor:
        mask = torch.zeros(tgt_len, tgt_len, dtype=torch.bool)
        for i in range(tgt_len):
            for j in range(1, i +1):
                mask[i][j] = False
            for j in range(i+1, tgt_len):
                mask[i][j] = True
        return mask


    def collate_translation(self, batch: list[tuple[str, str]], pad_id: int = 0) -> dict[str, torch.Tensor]:
        output = {}
        src_texts = [sentences[0] for sentences in batch]
        tgt_texts = [sentences[1] for sentences in batch]

        src_texts = [self.tokenizer.encode(sentence) for sentence in src_texts]
        pairs = [self.tokenizer.encode_tgt(sentence) for sentence in tgt_texts]
        tgt_in_texts, tgt_out_texts = zip(*pairs)
        tgt_in_texts, tgt_out_texts = list(tgt_in_texts), list(tgt_out_texts)

        src_texts = torch.tensor(self.pad_right(src_texts, pad_id), dtype=torch.long)
        tgt_in_texts = torch.tensor(self.pad_right(tgt_in_texts, pad_id), dtype=torch.long)
        tgt_out_texts = torch.tensor(self.pad_right(tgt_out_texts, pad_id), dtype=torch.long)

        src_pad_mask = (src_texts == pad_id)
        tgt_pad_mask = (tgt_in_texts == pad_id)
        tgt_causal_mask = self.make_triangle_mask(len(tgt_in_texts[0]))

        output["src_texts"] = src_texts
        output["tgt_in_texts"] = tgt_in_texts
        output["tgt_out_texts"] = tgt_out_texts
        output["src_pad_mask"] = src_pad_mask
        output["tgt_pad_mask"] = tgt_pad_mask
        output["tgt_causal_mask"] = tgt_causal_mask

        return output
