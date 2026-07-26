@echo off
setlocal EnableExtensions

set "APP_ROOT=%~dp0"
set "PYTHON=%APP_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo [CodeInsight] Virtual environment is missing.
  echo Run: .venv\Scripts\uv.exe sync --dev --locked
  exit /b 1
)

echo [CodeInsight] Stopping an existing CodeInsight server, if any...
powershell -NoProfile -Command "$root=[regex]::Escape('%APP_ROOT%'); $targets=@(); foreach($item in (Get-CimInstance Win32_Process)) { if($item.CommandLine -match $root -and $item.CommandLine -match 'uvicorn(\.exe)? .*api\.app:app' -and $item.CommandLine -match '--port 8000') { $targets += $item } }; foreach($target in $targets) { Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue }; Start-Sleep -Milliseconds 750"

echo [CodeInsight] Starting http://127.0.0.1:8000 ...
start "CodeInsight API" /D "%APP_ROOT%" "%PYTHON%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000

echo [CodeInsight] API documentation: http://127.0.0.1:8000/api/docs
echo [CodeInsight] Langfuse: http://localhost:3000
endlocal
