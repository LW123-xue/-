import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 当前地区名称
current_area_name_selector = '.bg_custom_1'
table_selector = '.table-sm'


# url: 要下载天气数据的地址
def get_weather_csv(url):
    options = webdriver.EdgeOptions()
    # options.add_argument('--headless')  # 无头模式
    service = webdriver.EdgeService(executable_path='C:/Users/24840/PycharmProjects/PythonProject/.venv/爬虫/msedgedriver.exe')
    driver = webdriver.Edge(service=service, options=options)

    driver.get(url)

    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, table_selector)))

    table = driver.find_element(By.CSS_SELECTOR, table_selector)

    data = {}

    head = table.find_element(By.CSS_SELECTOR, 'thead tr')
    th_list = head.find_elements(By.CSS_SELECTOR, 'th')
    for th in th_list:
        data[th.text] = []

    tr_list = table.find_elements(By.CSS_SELECTOR, 'tbody tr')

    for tr in tr_list:
        td_list = tr.find_elements(By.CSS_SELECTOR, 'td')
        for th, td in zip(th_list, td_list):
            data[th.text].append(td.text)

    # 查询当前城市的名称
    file_name = driver.find_element(By.CSS_SELECTOR, current_area_name_selector).text

    driver.quit()

    df = pd.DataFrame(data)

    # df.to_csv(f'src_data/{file_name}.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    get_weather_csv('https://datashareclub.com/weather/%E9%87%8D%E5%BA%86/%E9%87%8D%E5%BA%86/101040100.html')
