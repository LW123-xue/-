from backbones.transformer import ChatNet
import torch
from dataset.dataloader import generate_vocab
from dataset.vocab import tokenizer
import re

# 判断是否包含中文
def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

#预测
if __name__ == '__main__':
    vocab_zh, vocab_en = generate_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("正在加载模型...")
    model_zh2en = torch.load("./save/best.pt", weights_only=False)
    model_zh2en.eval()
    model_zh2en.to(device)

    model_en2zh = torch.load("./save/best_en2zh.pt", weights_only=False)
    model_en2zh.eval()
    model_en2zh.to(device)

    print("翻译机已启动（中↔英互译），输入 q 退出\n")

    while True:
        sentence = input("用户: ").strip()
        if sentence == "q":
            break

        # 自动判断翻译方向
        if is_chinese(sentence):
            print("AI（中→英）:", end=" ")
            model = model_zh2en
            src_vocab = vocab_zh
            tgt_vocab = vocab_en
            tokens = tokenizer([sentence], mode="char")[0]
        else:
            print("AI（英→中）:", end=" ")
            model = model_en2zh
            src_vocab = vocab_en
            tgt_vocab = vocab_zh
            tokens = tokenizer([sentence], mode="word")[0]

        # 未知词填充 0
        idx_list = []
        for t in tokens:
            idx = src_vocab.to_idx(t)
            idx_list.append(idx if idx is not None else 0)
        encode_idx = torch.tensor([idx_list]).to(device)

        # 初始化输出
        outputs = torch.tensor([[tgt_vocab.to_idx("<bos>")]]).to(device)
        result = []
        with torch.no_grad():
            for _ in range(20):
                results = model(encode_idx, outputs, None, None)
                idx = torch.argmax(results[:, -1], dim=-1)
                word = tgt_vocab[idx.item()]

                # 停止规则：解决无限句号+乱输出
                if word in ["<eos>", "<pad>", "<bos>", "<unk>"]:
                    break
                if not is_chinese(sentence) and word == "。":
                    result.append(word)
                    break

                result.append(word)
                outputs = torch.cat([outputs, idx.unsqueeze(0)], dim=-1)

        # 输出结果
        final_output = ' '.join(result) if is_chinese(sentence) else ''.join(result)
        print(f"{final_output}\n")