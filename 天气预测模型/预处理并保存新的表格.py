import os
import re

import pandas as pd

# 读取天气分类
with open('data/weathers.txt', 'r', encoding='utf-8') as f:
    weather_classes = f.read().split('\n')

# 读取风速分类
with open('data/speeds.txt', 'r', encoding='utf-8') as f:
    speed_classes = f.read().split('\n')

# 读取风向分类
with open('data/dirs.txt', 'r', encoding='utf-8') as f:
    dir_classes = f.read().split('\n')

# 读取所有的 csv，并保存为 DataFrame 列表
for entry in os.scandir('C:/Users/24840/PycharmProjects/PythonProject/.venv/爬虫/src_data'):
    df = pd.read_csv(entry.path, encoding='utf-8')
    # 按日期进行升序排序
    df = df.sort_values(by='日期')

    # 新的表格数据
    data = {}

    # 提取天气字段
    weathers = df['天气'].tolist()
    weather_idx = [weather_classes.index(weather) for weather in weathers]
    data['weather'] = weather_idx

    # 提取风向和风速
    # 解析风向和风速
    wind_dir_speed = df['风向/风速'].tolist()

    dirs = []
    speeds = []
    for item in wind_dir_speed:
        dir, speed = item.split()
        speeds.append(speed)
        dirs.append(dir)

    dirs = [dir_classes.index(dir) for dir in dirs]
    speeds = [speed_classes.index(speed) for speed in speeds]

    data['dir'] = dirs
    data['speed'] = speeds

    # 提取体感温度
    # 获取体感温度
    feels_like = df['体感温度'].tolist()

    regex = re.compile(r'^(-?\d+(\.\d+)?)℃$')

    feels = []
    for feel in feels_like:
        match = regex.search(feel)
        groups = match.groups()
        feels.append(float(groups[0]))
    data['feel'] = feels

    # 保存csv文件
    df = pd.DataFrame(data)
    df.to_csv(f'data/tgt_data/{entry.name}', index=False, encoding='utf-8')
