from multiprocessing import Process

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from 爬取指定网页的天气 import get_weather_csv

# 一级目录名称
level1_name = '重庆'

# 二级目录名称
level2_name = '重庆'

# 一个批次下载多少个城市的天气
batch_size = 5

# 选择器
# 地区列表
area_list_selector = '.scrollable>a'

options = webdriver.EdgeOptions()
service = webdriver.EdgeService(executable_path='C:/Users/24840/PycharmProjects/PythonProject/.venv/爬虫/msedgedriver.exe')
driver = webdriver.Edge(service=service, options=options)

# 拼接访问连接
url = f'https://datashareclub.com/area/{level1_name}/{level2_name}.html'

driver.get(url)

# 先获取要下载的地区列表
wait = WebDriverWait(driver, 30)
wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, area_list_selector)))

area_list = driver.find_elements(By.CSS_SELECTOR, area_list_selector)
# 获取 url 链接
urls = [area.get_attribute('href') for area in area_list]

driver.quit()

# 循环调用 get_weather_csv
# for url in urls:
#     get_weather_csv(url)

if __name__ == '__main__':
    # 多进程爬取
    for i in range(0, len(urls), batch_size):
        # 获取一个批次的 url
        _urls = urls[i:i + batch_size]
        ps_list = []
        for url in _urls:
            # 使用子进程
            process = Process(target=get_weather_csv, args=[url])
            process.start()
            ps_list.append(process)
        # 等待这个批次下载完，我们才开始下一个批次
        for process in ps_list:
            process.join()
