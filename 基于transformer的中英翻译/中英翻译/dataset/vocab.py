import re
import collections


# 读取文本
def read_txt(path="./data/timemachine.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()  # 按行读取文本内容
    return [re.sub("[^a-zA-Z]+", " ", line).lower().strip() for line in lines]


#分词
def tokenizer(lines, mode="word"):
    if mode == "word":
        return [line.split() for line in lines if len(line) >= 1]
    elif mode == "char":
        return [list(line) for line in lines]


#词频统计
def count_corpus(tokens):
    corpus_freq = collections.Counter([token for line in tokens for token in line])
    return list(corpus_freq.items())


#对齐句子长度，方便批次训练
def read_tokens(tokens):
    # 求出所有序列中最长的长度
    max_len = max([len(line) for line in tokens])
    for line in tokens:
        while len(line) < max_len:
            line += ['<pad>']
    return tokens

# 将人可读的文本（Token/单词）与机器可处理的数字（Index/索引）进行相互映射
class Vocab:
    def __init__(self, tokens, mini_freq=1, retired_tokens=None):
        super().__init__()
        assert len(tokens) > 0

        if retired_tokens is None:
            retired_tokens = ['<unk>', '<pad>']

        # <unk> 代表该词是未知的
        self.idx_to_token = []

        token_freqs = count_corpus(tokens)
        freq_sorted = sorted(token_freqs, key=lambda x: x[1], reverse=True)
        freq_sorted = [item for item in freq_sorted if item[1] > mini_freq]
        self.idx_to_token += retired_tokens
        self.idx_to_token += [item[0] for item in freq_sorted]
        self.token_to_idx = {item: index for index, item in enumerate(self.idx_to_token)}

    def __getitem__(self, indexes):  # 根据输入的id返回token
        if isinstance(indexes, (tuple | list)):
            return [self.idx_to_token[index] for index in indexes]
        return self.idx_to_token[indexes]

    def to_idx(self, tokens):
        if isinstance(tokens, (tuple | list)):
            return [self.token_to_idx.get(token, self.unk()) for token in tokens]
        return self.token_to_idx.get(tokens, 0)

    def __len__(self):
        return len(self.idx_to_token)

    def unk(self):
        return self.token_to_idx.get('<unk>')
