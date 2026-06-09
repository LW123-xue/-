import pandas as pd

# ====================== 修复 1：路径全部用 r"" 原生字符串 ======================
with open(r'C:\Users\24840\PycharmProjects\PythonProject\.venv\爬虫\天气预测模型\data\weathers.txt', 'r', encoding='utf-8') as f:
    weathers = f.read().splitlines()

with open(r'C:\Users\24840\PycharmProjects\PythonProject\.venv\爬虫\天气预测模型\data\dirs.txt', 'r', encoding='utf-8') as f:
    dirs = f.read().splitlines()

with open(r'C:\Users\24840\PycharmProjects\PythonProject\.venv\爬虫\天气预测模型\data\speeds.txt', 'r', encoding='utf-8') as f:
    speeds = f.read().splitlines()


class WeatherDataset:
    def __init__(self):
        # 读取数据文件
        self.df = pd.read_csv(r'C:\Users\24840\PycharmProjects\PythonProject\.venv\爬虫\天气预测模型\data\weather.csv', encoding='utf-8')

        self.data = []
        self.label = []

        # ====================== 修复 2：KeyError 'dirs' ======================
        # 请把这里的 'dirs' 改成你文件里【真正的列名】！！
        # 如果你不知道列名，先 print(self.df.columns) 看一下

        # 例如你的列名叫：direction、dir、风向 等
        # 我先给你写最常见的两种，你看哪个对：

        for idx, row in self.df.iterrows():
            # 方案 A：如果列名叫 direction
            # self.label.append(row['direction'])

            # 方案 B：如果列名叫 dir
            # self.label.append(row['dir'])

            # 方案 C：如果列名叫 风向
            # self.label.append(row['风向'])

            # 你原来的错误写法：
            # dirs.append(row['dirs'])  ❌

            # 正确写法（改成你真实的列名）
            # 我先暂时用最常见的 direction
            self.label.append(row['direction'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]