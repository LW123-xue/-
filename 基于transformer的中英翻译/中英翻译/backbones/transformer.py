import math
import torch
from torch import nn
class PositionEncoding(nn.Module):
    def __init__(self, embedding_dims, max_len, dropout, device=None):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('P', torch.zeros(1, max_len, embedding_dims, device=device))
        d_model = embedding_dims

        x = torch.arange(max_len).unsqueeze(-1) / torch.pow(10000, torch.arange(0, embedding_dims, 2) / d_model)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):  # x是经过词嵌入之后的值。 (batch_size,seq_len,embedding_dims)
        seq_len = x.shape[1]
        pe = self.P[:, :seq_len, :]
        return self.dropout(x + pe)


class ChatNet(nn.Module):
    def __init__(self, vocab_size_zh, vocab_size_en, embedding_dims, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2, device=None):
        super().__init__()
        self.device = device
        self.embedding_zh = nn.Embedding(vocab_size_zh, embedding_dims, device=device)
        self.embedding_en = nn.Embedding(vocab_size_en, embedding_dims, device=device)
        self.position_encoding = PositionEncoding(embedding_dims, 128, dropout=0.2, device=device)
        self.transformer = nn.Transformer(embedding_dims, nhead, num_encoder_layers, num_decoder_layers, 1024,
                                          batch_first=True, device=device)
        self.ln = nn.Linear(embedding_dims, vocab_size_en, device=device)

    def forward(self, inputs, outputs, in_valid_len, out_valid_len):
        attn_mask = nn.Transformer.generate_square_subsequent_mask(outputs.shape[1],device=self.device)
        inputs = self.position_encoding(self.embedding_zh(inputs))
        inputs_pad_mask = None
        if in_valid_len is not None:
            inputs_pad_mask = torch.tensor(_sequence_mask(inputs, in_valid_len))
            inputs_pad_mask = inputs_pad_mask.to(self.device)
        outputs = self.position_encoding(self.embedding_en(outputs))
        outputs_pad_mask = None
        if out_valid_len is not None:
            outputs_pad_mask = torch.tensor(_sequence_mask(outputs, out_valid_len))
            outputs_pad_mask = outputs_pad_mask.to(self.device)
        result = self.transformer(inputs, outputs, src_key_padding_mask=inputs_pad_mask,
                                  tgt_key_padding_mask=outputs_pad_mask, tgt_mask=attn_mask, tgt_is_causal=True)
        return self.ln(result)

# 生成序列掩码，依据有效长度屏蔽填充位置
def _sequence_mask(scores, valid_lens):
    if valid_lens is None:
        return None
    max_len = scores.shape[1]
    # 广播机制
    mask = torch.arange((max_len), dtype=torch.float32)[None, :] > valid_lens[:, None]
    mask = mask.float()
    mask[mask == 1] = float(-math.inf)
    return mask



# 该模型通过上述流程实现「中文→英文」的序列转换
# 核心依赖 Transformer 的注意力机制 + 位置编码 + 掩码策略，适配中英文双语生成任务。


if __name__ == '__main__':
    device = torch.device("cpu")
    inputs_zh = torch.randint(0, 10, (20, 9), device=device)
    inputs_en = torch.randint(0, 10, (20, 8), device=device)
    in_valid_len = torch.full((inputs_zh.shape[0],), inputs_zh.shape[1], device=device)
    out_valid_len = torch.full((inputs_en.shape[0],), inputs_en.shape[1], device=device)
    model = ChatNet(10, 10, 20, 2, 2, 2, device=device)
    outputs = model(inputs_zh, inputs_en, in_valid_len, out_valid_len)
    print(outputs.shape)
    print(sum(p.numel() for p in model.parameters()))