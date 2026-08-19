@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Local Knowledge Base (React + FastAPI)

REM ---------------------------------------------------------------
REM  IMPORTANT: keep this file 100%% ASCII.
REM  cmd.exe parses batch files byte by byte using the console code
REM  page, so multi-byte characters (e.g. Chinese) corrupt parsing
REM  in a way that depends on the machine's locale - it may work on
REM  one PC and fail on another. chcp 65001 does NOT fix this; it
REM  only changes console output, not how this file is parsed.
REM
REM  PYTHONUTF8 / PYTHONIOENCODING are required because this project
REM  may live under a path containing non-ASCII characters; without
REM  them pip and Python raise UnicodeEncodeError on cp950/cp1252.
REM
REM  Single process: the frontend is built to frontend\dist and then
REM  served by FastAPI as static files. No CORS, no second server.
REM  A prebuilt dist ships with this project, so Node.js is only
REM  needed if you want to MODIFY the frontend.
REM ---------------------------------------------------------------

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "APP_PORT=8600"
set "MARKER=%~dp0venv\.installed"
set "VENV_PY=%~dp0venv\Scripts\python.exe"

REM  Jump to INITDB, not RUN. The marker only means "packages are installed";
REM  it says nothing about the database. Jumping straight to RUN meant that
REM  anyone who deleted knowledge.db on an already-installed machine got the
REM  tables recreated by the backend but NO accounts, so nobody could log in
REM  and the cause was invisible. INITDB re-checks knowledge.db and is a no-op
REM  when it already exists.
if exist "%MARKER%" goto INITDB

echo.
echo ============================================================
echo   First run - setting up. This takes a few minutes.
echo   Later runs skip this and start immediately.
echo ============================================================
echo.

echo   [1/4] Locating Python ...
set "BASE_PY="
py -3.13 --version >nul 2>&1 && set "BASE_PY=py -3.13"
if not defined BASE_PY py -3.11 --version >nul 2>&1 && set "BASE_PY=py -3.11"
if not defined BASE_PY python --version >nul 2>&1 && set "BASE_PY=python"
if not defined BASE_PY (
  echo.
  echo   ERROR: Python 3.11 or newer was not found.
  echo   Install it from https://www.python.org/downloads/
  echo   and be sure to tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

echo   [2/4] Creating virtual environment ...
if not exist "%VENV_PY%" %BASE_PY% -m venv "%~dp0venv"
if not exist "%VENV_PY%" (
  echo   ERROR: failed to create the virtual environment.
  pause
  exit /b 1
)

echo   [3/4] Installing Python packages (this is the slow part) ...
"%VENV_PY%" -m pip install --upgrade pip --disable-pip-version-check --no-color --progress-bar off -q
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt" --disable-pip-version-check --no-color --progress-bar off
if errorlevel 1 (
  echo.
  echo   ERROR: package installation failed. Check your network / proxy.
  pause
  exit /b 1
)

REM  The frontend ships prebuilt. Only build it when dist is missing,
REM  so that people who just want to RUN the app never need Node.js.
echo   [4/4] Checking frontend ...
if exist "%~dp0frontend\dist\index.html" (
  echo         Prebuilt frontend found - skipping npm.
  goto DONE_SETUP
)

echo         No prebuilt frontend. Building from source ...
where npm >nul 2>&1
if errorlevel 1 (
  echo.
  echo   ERROR: frontend\dist is missing and Node.js / npm was not found.
  echo   Install Node.js 18+ from https://nodejs.org/ and run this again.
  echo.
  pause
  exit /b 1
)
pushd "%~dp0frontend"
call npm install --no-audit --no-fund
if errorlevel 1 ( popd & echo   ERROR: npm install failed. & pause & exit /b 1 )
call npm run build
if errorlevel 1 ( popd & echo   ERROR: frontend build failed. & pause & exit /b 1 )
popd

:DONE_SETUP
echo done > "%MARKER%"

:INITDB
REM  Create the database and demo accounts on first run.
REM
REM  knowledge.db is NOT in version control - it holds the full parsed text of
REM  every indexed document, which must never be published. The backend creates
REM  the tables on startup, but NOT the accounts, so without this step a fresh
REM  clone starts with zero users and nobody can log in.
REM
REM  No sample documents are created: this system is domain-agnostic and you
REM  point it at your own folder. Run "seed_data.py --with-demo-docs" if you
REM  want the bundled demo set.
if exist "%~dp0knowledge.db" goto RUN
echo.
echo   First run - creating database and demo accounts ...
"%VENV_PY%" "%~dp0seed_data.py"
if errorlevel 1 (
  echo   ERROR: database initialisation failed.
  pause
  exit /b 1
)

echo.
echo   Setup complete.
echo.

:RUN
echo.
echo ============================================================
echo   Local Knowledge Base
echo ------------------------------------------------------------
echo   URL       http://localhost:%APP_PORT%
echo   Accounts  admin / user01 / user02      password: demo1234
echo   Stop      press Ctrl+C in this window
echo ============================================================

REM  Keep the model names below in sync with the table in README.md. The two
REM  that used to be listed here were both on the "do not use" list:
REM  gemma4:e2b returns blank answers on follow-ups, and bge-large-zh-v1.5 is
REM  Chinese-only with a 512 context, which missed English documents entirely.
REM
REM  Comments must stay OUTSIDE the if-block below: cmd.exe parses parenthesised
REM  blocks as one unit and REM lines inside them are a known source of breakage.
REM
REM  Ollama powers the AI answering. The app still starts without it
REM  (browsing and admin work fine); the sidebar shows engine status.
curl -s -m 2 -o nul http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
  echo.
  echo   NOTE: Ollama does not seem to be running on port 11434.
  echo   AI answering will be unavailable until you start it.
  echo   Install: https://ollama.com/download
  echo   Models:  ollama pull bge-m3
  echo            ollama pull gemma4:12b
  echo.
)

start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 6; Start-Process 'http://localhost:%APP_PORT%'"
"%VENV_PY%" -m uvicorn backend.main:app --host 0.0.0.0 --port %APP_PORT%
