import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from .vocab import Vocab, tokenizer


df = pd.read_csv("../data/lightweight_zh2en_news.csv")
data = df.to_numpy()

out_tokens = tokenizer(data[:, 1], mode="word")
in_tokens = tokenizer(data[:, 0], mode="char")
vocab_zh = Vocab(in_tokens, 0, retired_tokens=['<pad>', '<bos>', '<eos>'])
vocab_en = Vocab(out_tokens, 0, retired_tokens=['<pad>', '<bos>', '<eos>'])
bos = [vocab_en.to_idx('<bos>')]
eos = [vocab_en.to_idx('<eos>')]
out_idx = [torch.tensor(bos + vocab_en.to_idx(line) + eos) for line in out_tokens]
in_idx = [torch.tensor(vocab_zh.to_idx(line)) for line in in_tokens]

# 自动构建并返回中英双语词表
def generate_vocab():
    return vocab_zh, vocab_en

# 通过索引读取单条样本，为训练提供输入数据
class TranslateDataset(Dataset):
    def __init__(self, sentence, translate_sentence):
        super().__init__()
        self.sentence = sentence
        self.translate_sentence = translate_sentence

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        return self.sentence[index], self.translate_sentence[index]


# 将同一批次数据全部对齐
def collate_fn(batch):
    in_inputs, out_inputs = zip(*batch)
    in_valid_len = torch.tensor([len(line) for line in in_inputs])
    out_valid_len = torch.tensor([len(line) for line in out_inputs])
    in_pad = pad_sequence(in_inputs, batch_first=True, padding_value=vocab_zh.to_idx("<pad>"))
    out_pad = pad_sequence(out_inputs, batch_first=True, padding_value=vocab_en.to_idx("<pad>"))

    return in_pad, out_pad, in_valid_len, out_valid_len


def generate_loader(batch_size=20):
    dataset = TranslateDataset(in_idx, out_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


