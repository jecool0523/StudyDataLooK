import argparse
import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


sys.stdout.reconfigure(encoding="utf-8")


DC_HOST = "https://gall.dcinside.com"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36"
)
REQUEST_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Referer": "https://gall.dcinside.com/",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}
CSV_COLUMNS = [
    "페이지",
    "게시글번호",
    "말머리",
    "제목",
    "내용",
    "글쓴이",
    "추천수",
    "조회수",
    "날짜",
    "링크",
]
THREAD_LOCAL = threading.local()


def parse_args():
    parser = argparse.ArgumentParser(
        description="디시인사이드 갤러리 게시글을 크롤링해 CSV로 저장합니다."
    )
    parser.add_argument("--gallery-id", required=True, help="갤러리 id 예: bugilacademy")
    parser.add_argument(
        "--gallery-type",
        choices=["mgallery", "mini"],
        default="mgallery",
        help="마이너갤러리는 mgallery, 미니갤러리는 mini",
    )
    parser.add_argument("--start-page", type=int, default=1, help="시작 페이지")
    parser.add_argument("--end-page", type=int, default=1, help="끝 페이지")
    parser.add_argument("--output", default=None, help="저장할 CSV 경로")
    parser.add_argument("--chromedriver", default=None, help="chromedriver.exe 경로")
    parser.add_argument("--headless", action="store_true", help="브라우저 창 없이 실행")
    parser.add_argument("--delay", type=float, default=0.05, help="페이지 사이 대기 시간(초)")
    parser.add_argument("--wait", type=int, default=5, help="Selenium 요소 대기 시간(초)")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="requests 방식에서 상세글을 동시에 가져올 작업자 수",
    )
    parser.add_argument(
        "--backend",
        choices=["requests", "selenium"],
        default="requests",
        help="크롤링 방식",
    )
    return parser.parse_args()


def gallery_base_url(gallery_type):
    return f"{DC_HOST}/{gallery_type}/board"


def list_url(gallery_id, page, gallery_type="mgallery"):
    query = urlencode({"id": gallery_id, "page": page})
    return f"{gallery_base_url(gallery_type)}/lists/?{query}"


def view_path_marker(gallery_type):
    return f"/{gallery_type}/board/view/"


def create_driver(chromedriver_path=None, headless=False):
    chrome_options = Options()
    chrome_options.page_load_strategy = "eager"
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument(f"--user-agent={DEFAULT_USER_AGENT}")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,
        },
    )

    if headless:
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1400,1000")

    if chromedriver_path:
        service = Service(chromedriver_path)
        return webdriver.Chrome(service=service, options=chrome_options)

    return webdriver.Chrome(options=chrome_options)


def get_thread_session():
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(REQUEST_HEADERS)
        THREAD_LOCAL.session = session
    return session


def make_requests_session(cookies=None):
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    if cookies:
        session.cookies.update(cookies)
    return session


def get_browser_cookies(
    gallery_id,
    gallery_type="mgallery",
    chromedriver=None,
    headless=True,
    wait_seconds=5,
):
    driver = create_driver(chromedriver, headless)
    try:
        driver.get(DC_HOST)
        time.sleep(1)
        driver.get(list_url(gallery_id, 1, gallery_type))
        time.sleep(wait_seconds)

        cookies = {}
        for cookie in driver.get_cookies():
            domain = cookie.get("domain", "")
            if "dcinside.com" in domain:
                cookies[cookie["name"]] = cookie["value"]
        return cookies
    finally:
        driver.quit()


def normalize_date(date_raw):
    date_txt = date_raw.strip().replace("/", ".")
    date_txt = re.sub(r"\s+", " ", date_txt)

    if re.fullmatch(r"\d{1,2}:\d{2}", date_txt):
        return datetime.now().strftime("%Y.%m.%d") + f" {date_txt}"
    if re.fullmatch(r"\d{2}\.\d{1,2}\.\d{1,2}.*", date_txt):
        return "20" + date_txt
    if re.fullmatch(r"\d{1,2}\.\d{1,2}", date_txt):
        return f"{datetime.now().year}.{date_txt}"
    return date_txt


def number_from_text(text):
    match = re.search(r"-?\d+", text.replace(",", ""))
    return int(match.group()) if match else None


def text_from_soup(node, selector):
    found = node.select_one(selector)
    return found.get_text(" ", strip=True) if found else ""


def absolute_dc_url(href):
    if not href:
        return ""
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return DC_HOST + href
    return href


def get_post_no(link):
    parsed = urlparse(link)
    return parse_qs(parsed.query).get("no", [""])[0]


def collect_post_links_fast(session, gallery_id, page, gallery_type="mgallery"):
    url = list_url(gallery_id, page, gallery_type)
    response_text = get_text_with_retry(session, url)
    soup = BeautifulSoup(response_text, "lxml")

    links = []
    for row in soup.select("tr.ub-content"):
        post_no = text_from_soup(row, ".gall_num")
        if not post_no.isdigit():
            continue

        title_link = row.select_one("td.gall_tit a[href]")
        if not title_link:
            continue

        link = absolute_dc_url(title_link.get("href"))
        if view_path_marker(gallery_type) in link:
            links.append(link)

    return links


def scrape_post_fast(session, link):
    response_text = get_text_with_retry(session, link)
    soup = BeautifulSoup(response_text, "lxml")

    author = text_from_soup(soup, ".gall_writer") or "익명"
    author = author.split("Image:")[0].strip()

    return {
        "페이지": "",
        "게시글번호": get_post_no(link),
        "말머리": text_from_soup(soup, ".title_headtext"),
        "제목": text_from_soup(soup, ".title_subject"),
        "내용": text_from_soup(soup, ".write_div"),
        "글쓴이": author,
        "추천수": number_from_text(
            text_from_soup(soup, ".up_num")
            or text_from_soup(soup, "#recommend_view_up_num")
            or text_from_soup(soup, ".gall_reply_num")
        ),
        "조회수": number_from_text(text_from_soup(soup, ".gall_count")),
        "날짜": normalize_date(text_from_soup(soup, ".gall_date")),
        "링크": link,
    }


def scrape_post_fast_worker(link):
    return scrape_post_fast(get_thread_session(), link)


def get_text_with_retry(session, url, retries=4, timeout=20):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            if response.text.strip():
                return response.text
            last_error = RuntimeError("empty response body")
        except RequestException as exc:
            last_error = exc

        sleep_seconds = min(8, 0.8 * attempt)
        print(f"  [retry {attempt}/{retries}] {url} - {last_error}")
        time.sleep(sleep_seconds)

    raise RuntimeError(f"failed to fetch after {retries} retries: {url} ({last_error})")


def save_csv(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def load_existing_rows(output_path):
    output_path = Path(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return []

    with output_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    normalized = []
    for row in rows:
        normalized.append({column: row.get(column, "") for column in CSV_COLUMNS})
    return normalized


def infer_resume_start_page(rows, default_start_page):
    if not rows:
        return default_start_page

    page_numbers = []
    for row in rows:
        try:
            page = int(row.get("페이지") or "")
        except ValueError:
            continue
        if page > 0:
            page_numbers.append(page)

    if page_numbers:
        return max(default_start_page, max(page_numbers))

    # Legacy CSVs did not store page numbers. Revisit the last likely page so
    # duplicate links are skipped and only missing posts are filled.
    estimated_completed_pages = max(0, len(rows) // 50)
    return max(default_start_page, default_start_page + max(0, estimated_completed_pages - 1))


def get_text_or_empty(parent, by, selector):
    try:
        return parent.find_element(by, selector).text.strip()
    except NoSuchElementException:
        return ""


def get_first_text_or_empty(parent, selectors):
    for by, selector in selectors:
        text = get_text_or_empty(parent, by, selector)
        if text:
            return text
    return ""


def collect_post_links(driver, gallery_id, page, gallery_type="mgallery", wait_seconds=8):
    driver.get(list_url(gallery_id, page, gallery_type))

    wait = WebDriverWait(driver, wait_seconds)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tr.ub-content")))
    except TimeoutException:
        print(f"[경고] {page}페이지 게시글 목록을 찾지 못했습니다.")
        return []

    links = []
    for row in driver.find_elements(By.CSS_SELECTOR, "tr.ub-content"):
        post_no = get_text_or_empty(row, By.CLASS_NAME, "gall_num")
        if not post_no.isdigit():
            continue

        try:
            link = row.find_element(By.CSS_SELECTOR, "td.gall_tit a").get_attribute("href")
        except NoSuchElementException:
            continue

        if link and view_path_marker(gallery_type) in link:
            links.append(link)

    return links


def scrape_post(driver, link, wait_seconds=8):
    driver.get(link)
    wait = WebDriverWait(driver, wait_seconds)
    title = wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "title_subject"))
    ).text.strip()

    author = get_text_or_empty(driver, By.CLASS_NAME, "gall_writer") or "익명"
    author = author.split("Image:")[0].strip()

    return {
        "페이지": "",
        "게시글번호": get_post_no(link),
        "말머리": get_text_or_empty(driver, By.CLASS_NAME, "title_headtext"),
        "제목": title,
        "내용": get_text_or_empty(driver, By.CLASS_NAME, "write_div"),
        "글쓴이": author,
        "추천수": number_from_text(
            get_first_text_or_empty(
                driver,
                [
                    (By.CSS_SELECTOR, ".up_num"),
                    (By.ID, "recommend_view_up_num"),
                    (By.CLASS_NAME, "gall_reply_num"),
                ],
            )
        ),
        "조회수": number_from_text(get_text_or_empty(driver, By.CLASS_NAME, "gall_count")),
        "날짜": normalize_date(get_text_or_empty(driver, By.CLASS_NAME, "gall_date")),
        "링크": link,
    }


def run_gallery(
    gallery_id,
    gallery_type="mgallery",
    start_page=1,
    end_page=1,
    output_path=None,
    backend="requests",
    workers=16,
    delay=0.05,
    wait=5,
    chromedriver=None,
    headless=False,
    cookies=None,
    resume_existing=False,
):
    if end_page < start_page:
        raise ValueError("--end-page must be greater than or equal to --start-page")

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "dc" / f"{gallery_id}.csv"
    output_path = Path(output_path)

    data = load_existing_rows(output_path) if resume_existing else []
    seen_links = {row.get("링크", "") for row in data if row.get("링크")}
    resume_start_page = infer_resume_start_page(data, start_page)

    if resume_existing and data:
        print(
            f"[{gallery_id}] resume: existing_rows={len(data)}, "
            f"start_from_page={resume_start_page}"
        )

    if backend == "requests":
        session = make_requests_session(cookies)

        for page in range(resume_start_page, end_page + 1):
            print(f"[{gallery_id}] page {page}: list")
            links = collect_post_links_fast(session, gallery_id, page, gallery_type)
            links = [link for link in links if link not in seen_links]
            seen_links.update(links)
            print(f"[{gallery_id}] page {page}: {len(links)} posts")

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_link = {
                    executor.submit(scrape_post_fast_worker, link): link for link in links
                }

                for idx, future in enumerate(as_completed(future_to_link), start=1):
                    link = future_to_link[future]
                    try:
                        post = future.result()
                        post["페이지"] = page
                        data.append(post)
                        print(f"  [{idx}/{len(links)}] saved {post['게시글번호']}")
                    except Exception as exc:
                        print(f"  [failed] {link} - {exc}")

            save_csv(data, output_path)
            print(f"[{gallery_id}] checkpoint: {len(data)} rows")
            if delay:
                time.sleep(delay)
    else:
        driver = create_driver(chromedriver, headless)
        try:
            for page in range(resume_start_page, end_page + 1):
                print(f"[{gallery_id}] page {page}: list")
                links = collect_post_links(driver, gallery_id, page, gallery_type, wait)
                links = [link for link in links if link not in seen_links]
                seen_links.update(links)
                print(f"[{gallery_id}] page {page}: {len(links)} posts")

                for idx, link in enumerate(links, start=1):
                    try:
                        post = scrape_post(driver, link, wait)
                        post["페이지"] = page
                        data.append(post)
                        print(f"  [{idx}/{len(links)}] saved {post['게시글번호']}")
                    except Exception as exc:
                        print(f"  [failed] {link} - {exc}")
                    if delay:
                        time.sleep(delay)

                save_csv(data, output_path)
                print(f"[{gallery_id}] checkpoint: {len(data)} rows")
        finally:
            driver.quit()

    save_csv(data, output_path)
    print(f"[done] {gallery_id}: {len(data)} rows -> {output_path}")
    return data


def main():
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "dc" / f"{args.gallery_id}.csv"
    )
    run_gallery(
        gallery_id=args.gallery_id,
        gallery_type=args.gallery_type,
        start_page=args.start_page,
        end_page=args.end_page,
        output_path=output_path,
        backend=args.backend,
        workers=args.workers,
        delay=args.delay,
        wait=args.wait,
        chromedriver=args.chromedriver,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
