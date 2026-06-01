# DC School Gallery Crawler

This package crawls DCInside school galleries listed in `school_list.json`.
It includes the current partial CSV results, so another PC can continue only
the unfinished schools/pages.

## Requirements

- Windows
- Python 3.10+
- Google Chrome
- Internet connection

ChromeDriver does not need to be installed manually. Selenium Manager will try
to download the matching driver automatically when Selenium fallback is used.

## Quick Start

1. Unzip this folder.
2. Double-click `run_remaining_requests.bat`.
3. If many schools still fail with empty responses, wait a while and run
   `run_remaining_with_fallback.bat`.

Outputs are saved in:

```text
dc\schools_20p\
```

The combined CSV is:

```text
dc\schools_20p\all_schools.csv
```

## Recommended Commands

Fast requests-only resume:

```powershell
python batch_crawl_dc_schools.py --school-list school_list.json --pages 20 --parallel-galleries 1 --workers-per-gallery 4 --delay 1.5 --resume --min-rows-to-skip 800 --page-resume --bootstrap-cookies
```

Requests plus Selenium fallback:

```powershell
python batch_crawl_dc_schools.py --school-list school_list.json --pages 20 --parallel-galleries 1 --workers-per-gallery 3 --delay 2.0 --resume --min-rows-to-skip 800 --page-resume --bootstrap-cookies --fallback-selenium --fallback-min-rows 1
```

## How Resume Works

- Existing CSV rows are loaded first.
- Existing `링크` values are skipped to prevent duplicates.
- If a CSV has a `페이지` column, crawling resumes near the latest page.
- If a CSV is older and has no `페이지` values, the script estimates the last
  page from row count and rechecks that area.
- `--min-rows-to-skip 800` treats schools with at least 800 rows as sufficiently
  done for a 20-page run.

## If DCInside Returns Empty Responses

Reduce request pressure:

```powershell
python batch_crawl_dc_schools.py --school-list school_list.json --pages 20 --parallel-galleries 1 --workers-per-gallery 2 --delay 3.0 --resume --min-rows-to-skip 800 --page-resume --bootstrap-cookies
```

Using a different network/IP can also help.
