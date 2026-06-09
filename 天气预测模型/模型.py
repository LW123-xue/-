import torch
from torch import nn


class WeatherModel(nn.Module):
    def __init__(self, input_size=97, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_size * 2, input_size)

        # 定义不同的头
        # 天气头
        self.weather_head = nn.Linear(hidden_size * 2, 26)
        # self.weather_head = nn.Sequential(
        #     nn.Linear(hidden_size * 2, 256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, 26),
        # )
        # 风向头
        self.dir_head = nn.Linear(hidden_size * 2, 64)
        # 风速头
        self.speed_head = nn.Linear(hidden_size * 2, 6)
        # 温度头
        self.feel_head = nn.Linear(hidden_size * 2, 1)

    # x: (N, L, input_size)
    # h: (D=2 * num_layers=2, N, hidden_size)
    # return: (N, L, input_size)
    # 返回值的特征包含几个部分: 26个天气分类 + 64个风向分类 + 6个风速分类 + 1个温度
    def forward(self, x, h=None):
        x, h = self.gru(x, h)
        # x (N, L, D=2 * hidden_size): x 代表的含义：预测的平移 1 位后的天气特征

        # 沿着 最后一个维度 归一化
        m = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - m) / torch.sqrt(std ** 2 + 1e-6)


        # 得到每天的天气特征后，此处可以有两种做法
        # 方法一: 直接用全连接，一步到位预测出对应天气的特征数，此处为 97
        # return self.fc(x), h

        # 方法二: 使用不同的头去预测不同的特征
        # 头（head）: 模型将隐藏层的特征进行输出的层就是头
        # 分别调用不同的头
        weathers = self.weather_head(x)
        dirs = self.dir_head(x)
        speeds = self.speed_head(x)
        feels = self.feel_head(x)
        # 拼接结果
        y = torch.concat([weathers, dirs, speeds, feels], dim=-1)
        return y, h


if __name__ == '__main__':
    from 数据集 import WeatherDataset

    ds = WeatherDataset()
    inputs, labels = ds[0]
    inputs = inputs.unsqueeze(0)
    model = WeatherModel()
    y, h = model(inputs)
    print(y.shape)
    print(h.shape)
