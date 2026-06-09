import os
import re
import pandas as pd

df_list = [pd.read_csv(entry.path,encoding='utf-8') for entry in os.scandir('C:/Users/24840/PycharmProjects/PythonProject/.venv/爬虫/src_data')]
df = pd.concat(df_list)
weathers = df['天气']
weathers = weathers.drop_duplicates().sort_values()
weathers = weathers.tolist()
print(weathers)
with open('data/weathers.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(weathers))
wind_dir_speed = df['风向/风速'].tolist()

# 使用正则表达式，验证数据格式是否使用空格隔开风向和风速
regex = re.compile(r'^\S+ \S+$')
print(all([regex.search(item) for item in wind_dir_speed]))

dirs = []
speeds = []
for item in wind_dir_speed:
    dir, speed = item.split()
    speeds.append(speed)
    dirs.append(dir)

speeds = sorted(list(set(speeds)))
dirs = sorted(list(set(dirs)))

print(speeds)
with open('data/speeds.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(speeds))
src_dirs = ['东风', '东北风', '北风', '西北风', '西风', '西南风', '南风', '东南风']
print(f'src_dirs: {src_dirs}')

_dirs = []

for i in range(1, len(src_dirs)):
    # 滚动起始方向 src_dirs 获得滚动后的 target_dirs
    target_dirs = src_dirs[i:] + src_dirs[:i]
    print(f'target_dirs: {target_dirs}')
    # 循环每一组 src 和 target
    for src_dir, target_dir in zip(src_dirs, target_dirs):
        # 拼接风向
        _dirs.append(f'{src_dir}转{target_dir}')

# 拼接 src_dirs
_dirs = src_dirs + _dirs
print(_dirs)
print(len(_dirs))
with open('data/dirs.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(_dirs))

# 验证表格中的风向是否都从 dirs 中找到索引
for dir in dirs:
    print(_dirs.index(dir))

# 获取体感温度
feels_like = df['体感温度'].tolist()

regex = re.compile(r'^(-?\d+(\.\d+)?)℃$')

for feel in feels_like:
    match = regex.search(feel)
    groups = match.groups()
    # print(float(groups[0]))