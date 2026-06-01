$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$Python = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "[setup] Creating virtual environment..."
    $BasePython = $null
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        $BasePython = "python"
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $BasePython = "py -3"
    } else {
        throw "Python 3 is not installed or not on PATH. Install Python 3.10+ first."
    }

    Invoke-Expression "$BasePython -m venv .venv"
}

Write-Host "[setup] Installing dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

Write-Host "[run] Requests crawler: resume unfinished schools"
& $Python batch_crawl_dc_schools.py `
    --school-list "$ScriptDir\school_list.json" `
    --pages 20 `
    --parallel-galleries 1 `
    --workers-per-gallery 4 `
    --delay 1.5 `
    --resume `
    --min-rows-to-skip 800 `
    --page-resume `
    --bootstrap-cookies

Write-Host "[done] Output: $ScriptDir\dc\schools_20p"
