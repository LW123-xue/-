import torch
from torch import nn

from 模型 import WeatherModel
from 数据集 import WeatherDataset
from torch.utils.data import random_split, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 100
lr = 1e-3
batch_size = 64
weight_decay = 1e-4
train_ratio = 0.8  # 训练数据占比
valid_interval = 10  # 验证间隔，每训练 10 个 epoch 验证一次模型
checkpoint_path = 'checkpoints/checkpoint_10.pth'
start_epoch = 0

# 创建数据集
ds = WeatherDataset()
# 划分数据集
train_len = int(len(ds) * 0.8)
valid_len = len(ds) - train_len
train_ds, valid_ds = random_split(ds, [train_len, valid_len])
# 创建数据加载器
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

# 模型
model = WeatherModel()

# 优化器
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# 学习率调度器
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optim,
    T_max=EPOCH,
    eta_min=1e-7
)

# 加载检查点
try:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
except:
    print('加载检查点失败，使用默认参数开始训练')
model.to(device)

ce = nn.CrossEntropyLoss()
smooth_l1 = nn.SmoothL1Loss()


# 损失函数
def loss_fn(y, labels):
    # y (N, L, input_size=97) 26个天气分类 + 64个风向分类 + 6个风速分类 + 1个温度
    # labels (N, L, 4)
    # 天气损失
    loss1 = ce(y[..., :26].reshape(y.shape[0] * y.shape[1], -1), labels[..., 0].long().reshape(-1))
    # 风向损失
    loss2 = ce(y[..., 26:26 + 64].reshape(y.shape[0] * y.shape[1], -1), labels[..., 1].long().reshape(-1))
    # 风速损失
    loss3 = ce(y[..., 26 + 64:26 + 64 + 6].reshape(y.shape[0] * y.shape[1], -1), labels[..., 2].long().reshape(-1))
    # 温度损失
    loss4 = smooth_l1(y[..., -1], labels[..., 3])
    return loss1 + loss2 + loss3 + loss4


def train():
    model.train()

    total_loss = 0.
    count = 0
    total_grad = 0.

    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)

        optim.zero_grad()
        y, h = model(inputs)
        loss = loss_fn(y, labels)
        loss.backward()
        # 统计损失和梯度
        total_loss += loss.item()
        total_grad += sum([p.grad.norm().item() for p in model.parameters() if p.grad is not None])
        count += 1
        optim.step()

    return total_loss / count, total_grad, total_grad / count


def valid():
    model.eval()

    total_loss = 0.
    count = 0

    for inputs, labels in valid_dl:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            y, h = model(inputs)
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1

    return total_loss / count


best_valid_loss = valid()

for epoch in range(start_epoch, EPOCH):
    train_loss, total_grad, avg_grad = train()
    if (epoch + 1) % valid_interval == 0:
        valid_loss = valid()
        print(f'EPOCH: [{epoch + 1}/{EPOCH}]; Train Loss: {train_loss}; Valid Loss: {valid_loss}; Avg Grad: {avg_grad}')
        # 判断是否是最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # 保存最佳模型
            torch.save(model.state_dict(), 'weights/best_model.pth')

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_{epoch + 1}.pth')

    # 调用学习率调度器
    lr_scheduler.step()

# 保存最后训练完的模型
torch.save(model.state_dict(), 'weights/last_model.pth')
