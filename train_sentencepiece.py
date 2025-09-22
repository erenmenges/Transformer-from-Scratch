import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='/Users/erenmenges/Desktop/transformer/dataset/concat.txt',
    model_prefix='/Users/erenmenges/Desktop/transformer/tokenizer_model',
    vocab_size=10000,
    model_type='bpe',
    character_coverage=1.0,
    pad_id=0, unk_id=1, bos_id=2, eos_id=3
)