import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import sys
sys.stdout.reconfigure(encoding='utf-8')

def setup_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-gpu')
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def parse_list_page(driver, page_num):
    url = f'https://pann.nate.com/talk/c20038?page={page_num}'
    driver.get(url)
    time.sleep(1.0)
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')

    #rows = soup.select('table tbody tr')
    titles = soup.select('h2 > a')
    data = []

    for a in titles:
        if not a or not a.has_attr('href'):
            continue
        print(a['href'], a.text.strip())
        href = a['href']                   # e.g. '/talk/374648775'
        title = a.get_text(strip=True)

        writer = a.find_next_sibling('span', class_='writer')
        stats  = a.find_next_sibling(text=True)  # "<조회수> <시간>" 텍스트
        print("title_links 추출 수:", len(titles))  # → 실제 글 개수와 일치해야 정상
        if len(titles) == 0:
            print("셀렉터 오류: HTML 검사 도구 ↦ 'h2 > a' 구조 확인 필요")

        data.append({
            'view_url': f'https://pann.nate.com{href}',
            'title': title,
        })
    return data

def parse_view_page(driver, url):
    driver.get(url)
    time.sleep(0.7)
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')

    content_div = soup.select_one('div#postContent')
    content = content_div.get_text('\n', strip=True) if content_div else ''
    # 이미지 주소들 (예: <img src="…">)
    imgs = [img['src'] for img in content_div.select('img')] if content_div else []

    comment_cnt = soup.select_one('span.cmt_total > em')
    comments = int(comment_cnt.get_text()) if comment_cnt else 0

    return content, imgs, comments

def crawl(max_pages=4):
    driver = setup_driver(headless=True)
    all_posts = []
    for p in range(1, max_pages + 1):
        print(f'리스트 페이지 {p} 크롤링 중…')
        list_data = parse_list_page(driver, p)
        for post in list_data:
            print(f"    → {post['post_id']} 제목: {post['title']}")
            content, imgs, comments = parse_view_page(driver, post['view_url'])
            post['content'] = content
            post['images'] = imgs
            post['comment_cnt'] = comments
            all_posts.append(post)
        time.sleep(0.5)
    driver.quit()
    return pd.DataFrame(all_posts)

# 실행
if __name__ == '__main__':
    df = crawl(max_pages=2)
    df.to_csv('pann_10대이야기.csv', index=False, encoding='utf‑8-sig')
