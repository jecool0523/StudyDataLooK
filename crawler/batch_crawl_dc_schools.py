import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from crawler_mimi import CSV_COLUMNS, get_browser_cookies, run_gallery


sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_SCHOOL_LIST = str(Path(__file__).resolve().parent / "school_list.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="school_list.json의 디시 갤러리를 학교별로 자동 크롤링합니다."
    )
    parser.add_argument(
        "--school-list",
        default=DEFAULT_SCHOOL_LIST,
        help="학교 목록 JSON 경로",
    )
    parser.add_argument("--pages", type=int, default=20, help="학교별 크롤링 페이지 수")
    parser.add_argument("--start-page", type=int, default=1, help="시작 페이지")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "dc" / "schools_20p"),
        help="학교별 CSV 저장 폴더",
    )
    parser.add_argument(
        "--combined-output",
        default=None,
        help="전체 통합 CSV 경로. 생략하면 output-dir/all_schools.csv",
    )
    parser.add_argument(
        "--parallel-galleries",
        type=int,
        default=1,
        help="동시에 크롤링할 학교 수",
    )
    parser.add_argument(
        "--workers-per-gallery",
        type=int,
        default=16,
        help="학교 하나 안에서 상세글을 병렬 요청할 작업자 수",
    )
    parser.add_argument("--delay", type=float, default=0.05, help="페이지 사이 대기 시간")
    parser.add_argument(
        "--backend",
        choices=["requests", "selenium"],
        default="requests",
        help="크롤링 방식",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="학교별 CSV가 충분한 행 수로 이미 있으면 해당 학교를 건너뜁니다.",
    )
    parser.add_argument(
        "--min-rows-to-skip",
        type=int,
        default=None,
        help="resume 시 이 행 수 이상인 CSV만 완료로 간주합니다. 기본값은 pages*40",
    )
    parser.add_argument(
        "--bootstrap-cookies",
        action="store_true",
        help="Chrome으로 디시에 접속해 받은 쿠키를 requests 크롤링에 주입합니다.",
    )
    parser.add_argument(
        "--fallback-selenium",
        action="store_true",
        help="requests 방식이 실패하거나 너무 적게 저장되면 Selenium으로 재시도합니다.",
    )
    parser.add_argument(
        "--page-resume",
        action="store_true",
        help="기존 학교별 CSV를 읽어 중복 링크를 건너뛰고 남은 페이지부터 이어받습니다.",
    )
    parser.add_argument(
        "--reverse-schools",
        action="store_true",
        help="school_list.json의 뒤쪽 학교부터 처리합니다.",
    )
    parser.add_argument(
        "--exclude-ids",
        default="",
        help="크롤링에서 제외할 갤러리 id 목록. 쉼표로 구분합니다. 예: dimigo,abc",
    )
    parser.add_argument(
        "--fallback-min-rows",
        type=int,
        default=1,
        help="이 행 수 미만이면 Selenium fallback을 실행합니다.",
    )
    parser.add_argument("--chromedriver", default=None, help="chromedriver.exe 경로")
    parser.add_argument(
        "--cookie-wait",
        type=int,
        default=5,
        help="쿠키 수집을 위해 Chrome에서 목록 페이지를 열고 기다릴 시간(초)",
    )
    return parser.parse_args()


def load_schools(path):
    with open(path, "r", encoding="utf-8") as f:
        schools = json.load(f)

    if not isinstance(schools, list):
        raise ValueError("school_list.json must contain a list of schools")

    normalized = []
    for school in schools:
        name = str(school.get("name", "")).strip()
        gallery_id = str(school.get("id", "")).strip()
        if not gallery_id:
            continue

        normalized.append(
            {
                "name": name,
                "id": gallery_id,
                "gallery_type": infer_gallery_type(name),
            }
        )

    return normalized


def infer_gallery_type(name):
    if "ⓜ" in name:
        return "mgallery"
    if "ⓝ" in name:
        return "mini"
    return "mgallery"


def clean_school_name(name):
    name = name.replace("ⓜ", "").replace("ⓝ", "").strip()
    return name or "unknown"


def safe_filename(value):
    value = clean_school_name(value)
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value).strip("._ ")
    return value or "unknown"


def save_combined(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["학교명", "갤러리ID", "갤러리타입"] + CSV_COLUMNS
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def enrich_rows(rows, school):
    school_name = clean_school_name(school["name"])
    return [
        {
            "학교명": school_name,
            "갤러리ID": school["id"],
            "갤러리타입": school["gallery_type"],
            **row,
        }
        for row in rows
    ]


def read_school_csv(output_path, school):
    with Path(output_path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return enrich_rows(rows, school)


def read_row_count(output_path):
    with Path(output_path).open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def crawl_school(school, args, output_dir, cookies=None):
    school_name = clean_school_name(school["name"])
    gallery_id = school["id"]
    gallery_type = school["gallery_type"]
    filename = f"{safe_filename(school_name)}__{gallery_id}.csv"
    output_path = output_dir / filename
    min_rows_to_skip = (
        args.min_rows_to_skip if args.min_rows_to_skip is not None else args.pages * 40
    )

    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        existing_rows = read_row_count(output_path)
        if existing_rows >= min_rows_to_skip:
            print(
                f"[skip] {school_name} ({gallery_id}) rows={existing_rows}: {output_path}"
            )
            return read_school_csv(output_path, school)
        print(
            f"[rerun] {school_name} ({gallery_id}) rows={existing_rows} "
            f"< {min_rows_to_skip}"
        )

    print(f"[start] {school_name} ({gallery_type}/{gallery_id})")
    try:
        rows = run_gallery(
            gallery_id=gallery_id,
            gallery_type=gallery_type,
            start_page=args.start_page,
            end_page=args.start_page + args.pages - 1,
            output_path=output_path,
            backend=args.backend,
            workers=args.workers_per_gallery,
            delay=args.delay,
            headless=True,
            cookies=cookies,
            resume_existing=args.page_resume,
        )
    except Exception as exc:
        if not args.fallback_selenium or args.backend == "selenium":
            raise
        print(f"[fallback] {school_name} ({gallery_id}) requests failed: {exc}")
        rows = []

    if args.fallback_selenium and args.backend != "selenium" and len(rows) < args.fallback_min_rows:
        print(
            f"[fallback] {school_name} ({gallery_id}) rows={len(rows)} "
            f"< {args.fallback_min_rows}; retry with Selenium"
        )
        rows = run_gallery(
            gallery_id=gallery_id,
            gallery_type=gallery_type,
            start_page=args.start_page,
            end_page=args.start_page + args.pages - 1,
            output_path=output_path,
            backend="selenium",
            workers=args.workers_per_gallery,
            delay=max(args.delay, 0.5),
            wait=8,
            chromedriver=args.chromedriver,
            headless=True,
            resume_existing=args.page_resume,
        )

    print(f"[done-school] {school_name}: {len(rows)} rows")
    return enrich_rows(rows, school)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_output = (
        Path(args.combined_output)
        if args.combined_output
        else output_dir / "all_schools.csv"
    )

    schools = load_schools(args.school_list)
    exclude_ids = {item.strip() for item in args.exclude_ids.split(",") if item.strip()}
    if exclude_ids:
        before = len(schools)
        schools = [school for school in schools if school["id"] not in exclude_ids]
        print(f"[batch] excluded={sorted(exclude_ids)} ({before} -> {len(schools)})")

    if args.reverse_schools:
        schools = list(reversed(schools))

    cookies = None
    if args.bootstrap_cookies and schools:
        first = schools[0]
        print(
            f"[cookies] bootstrap with {first['gallery_type']}/{first['id']} "
            f"wait={args.cookie_wait}s"
        )
        cookies = get_browser_cookies(
            gallery_id=first["id"],
            gallery_type=first["gallery_type"],
            chromedriver=args.chromedriver,
            headless=True,
            wait_seconds=args.cookie_wait,
        )
        print(f"[cookies] collected={len(cookies)}")

    print(f"[batch] schools={len(schools)}, pages={args.pages}")
    print(f"[batch] output-dir={output_dir}")

    all_rows = []
    if args.parallel_galleries <= 1:
        for school in schools:
            try:
                all_rows.extend(crawl_school(school, args, output_dir, cookies))
                save_combined(all_rows, combined_output)
            except Exception as exc:
                print(f"[failed-school] {school['name']} ({school['id']}) - {exc}")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel_galleries) as executor:
            future_to_school = {
                executor.submit(crawl_school, school, args, output_dir, cookies): school
                for school in schools
            }
            for future in as_completed(future_to_school):
                school = future_to_school[future]
                try:
                    all_rows.extend(future.result())
                    save_combined(all_rows, combined_output)
                except Exception as exc:
                    print(f"[failed-school] {school['name']} ({school['id']}) - {exc}")

    save_combined(all_rows, combined_output)
    print(f"[done-batch] rows={len(all_rows)} -> {combined_output}")


if __name__ == "__main__":
    main()
