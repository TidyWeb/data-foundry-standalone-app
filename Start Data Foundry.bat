@echo off
rem Data Foundry launcher for Windows.
cd /d "%~dp0"
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if defined PY goto have_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto have_python
echo Data Foundry needs Python 3.13 or newer.
echo Download it from https://www.python.org/downloads/ and run this again.
pause
exit /b 1

:have_python
if exist venv\Scripts\python.exe goto run
echo First run: setting things up. This needs an internet connection and takes a minute or two.
%PY% -m venv venv
if errorlevel 1 goto failed
venv\Scripts\python.exe -m pip install --quiet -r requirements.txt
if errorlevel 1 goto failed

:run
for /f %%p in ('venv\Scripts\python.exe -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()"') do set PORT=%%p
echo Data Foundry is starting at http://127.0.0.1:%PORT%/
echo Leave this window open while you use it. Close it to stop.
start "" /b cmd /c "timeout /t 3 >nul & start http://127.0.0.1:%PORT%/"
venv\Scripts\python.exe -m flask --app app run --port %PORT%
pause
exit /b 0

:failed
echo Setup failed (see the messages above).
pause
exit /b 1
