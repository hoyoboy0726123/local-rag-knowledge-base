@echo off
chcp 65001 >nul
cd /d "%~dp0"
title RD PM Knowledge Base V2 (React + FastAPI)

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

if exist "%MARKER%" goto RUN

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
echo.
echo   Setup complete.
echo.

:RUN
echo.
echo ============================================================
echo   RD PM Knowledge Base V2
echo ------------------------------------------------------------
echo   URL       http://localhost:%APP_PORT%
echo   Accounts  admin / pm01 / pm02      password: demo1234
echo   Stop      press Ctrl+C in this window
echo ============================================================

REM  Ollama powers the AI answering. The app still starts without it
REM  (browsing and admin work fine); the sidebar shows engine status.
curl -s -m 2 -o nul http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
  echo.
  echo   NOTE: Ollama does not seem to be running on port 11434.
  echo   AI answering will be unavailable until you start it.
  echo   Install: https://ollama.com/download
  echo   Models:  ollama pull gemma4:e2b
  echo            ollama pull quentinz/bge-large-zh-v1.5
  echo.
)

start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 6; Start-Process 'http://localhost:%APP_PORT%'"
"%VENV_PY%" -m uvicorn backend.main:app --host 0.0.0.0 --port %APP_PORT%
