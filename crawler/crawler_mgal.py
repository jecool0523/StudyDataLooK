import sys
sys.stdout.reconfigure(encoding='utf-8')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# 크롬 옵션 설정
chrome_options = Options()
chrome_options.add_argument('--start-maximized')
chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36')

# chromedriver 경로 설정 (본인 PC에 맞게 수정하세요)
service = Service(r'C:\Users\seocheon\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe')

# 드라이버 실행
driver = webdriver.Chrome(service=service, options=chrome_options)

# 결과 저장 리스트
data = []

# 몇 페이지까지 크롤링할지 설정
for page in range(1, 20):
    url = f"https://gall.dcinside.com/mgallery/board/lists/?id=skk&page={page}"
    #"https://gall.dcinside.com/mini/board/lists/?id=phsfanclub&page={page}"
    #https://gall.dcinside.com/mini/board/view/?id=phsfanclub&no=191&page=2
    #https://gall.dcinside.com/mini/board/lists/?id=dimigo&page={page}
    #https://gall.dcinside.com/mini/board/view/?id=dimigo&no=4267&page=1
    #https://gall.dcinside.com/mgallery/board/lists/?id=bsis&page={page}
    #https://gall.dcinside.com/mgallery/board/view/?id=bsis&no=1787&page=1
    print(f"{page} 페이지 크롤링 중...")

    driver.get(url)
    time.sleep(1)

    #posts = driver.find_elements(By.CSS_SELECTOR, "tr.ub-content.us-post")
    # 1) posts 대신 links라는 리스트로 링크만 수집
    links = []
    all_tr = driver.find_elements(By.CSS_SELECTOR, "tr.ub-content")
    for tr in all_tr:
        try:
            gall_num = tr.find_element(By.CLASS_NAME, "gall_num").text.strip()
            if not gall_num.isdigit():
                continue
            link = tr.find_element(By.CSS_SELECTOR, "td.gall_tit a").get_attribute("href")
            links.append(link)
        except Exception:
            continue
    print(f"게시글 수: {len(links)}")

    # 2) links를 순회하면서 상세글 접근
    for link in links:
        try:
            driver.get(link)
            wait = WebDriverWait(driver, 5)
            title_detail = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "title_subject"))).text.strip()
            content = driver.find_element(By.CLASS_NAME, "write_div").text.strip()
            recommend = driver.find_element(By.CLASS_NAME, "gall_reply_num").text.strip()
            view_count = driver.find_element(By.CLASS_NAME, "gall_count").text.strip()
            from datetime import datetime

            date_raw = driver.find_element(By.CLASS_NAME, "gall_date").text.strip()
            date_txt = date_raw.replace("/", ".")
            parts = date_txt.split(".")
            if len(parts) == 2:
                year = str(datetime.now().year)[2:]
                date_txt = f"{year}.{parts[0]}.{parts[1]}"
            elif len(parts) == 3 and len(parts[0]) == 2:
                date_txt = f"20{parts[0]}.{parts[1]}.{parts[2]}"
            date = date_txt

            try:
                author_td = driver.find_element(By.CLASS_NAME, "gall_writer")
                author = author_td.text.strip()
                author = author.split("Image:")[0].strip()
            except:
                author = "익명"

            try:
                head = driver.find_element(By.CLASS_NAME, "title_headtext").text.strip()
            except:
                head = ""

            data.append({
                "제목": title_detail,
                "말머리": head,
                "내용": content,
                "글쓴이": author,
                "추천수": recommend,
                "조회수": view_count,
                "날짜": date,
                "링크": link
            })

        except Exception as e:
            print(f": {e}")
            continue  # 다음 글로 넘어감
        
driver.quit()

# 데이터 저장
df = pd.DataFrame(data)
df.to_csv("skk.csv", index=False, encoding="utf-8-sig")
print("크롤링 완료,.csv 저장됨.")