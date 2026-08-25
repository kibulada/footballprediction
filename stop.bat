@echo off
setlocal
cd /d "%~dp0"

REM ----------------------------------------------------------------
REM Stop the Hermes Football bot.
REM Kills every python.exe running bot.py (any interpreter: .venv or
REM uv-managed), then cleans leftover bot-spawned Chrome zombies
REM (headless / soccerdata / --host-resolver-rules) so the next start
REM is clean. Regular browsing Chrome is LEFT ALONE.
REM Single-instance lock (.bot.lock) is released automatically when
REM the process dies -- no manual cleanup needed.
REM ----------------------------------------------------------------
for /f "tokens=*" %%P in (
    'powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter 'Name=''python.exe''' | Where-Object { $_.CommandLine -like '*bot.py*' } | Select-Object -ExpandProperty ProcessId)"'
) do (
    if not "%%P"=="" (
        echo [stop.bat] stopping bot PID %%P
        taskkill /F /PID %%P >NUL 2>&1
    )
)

for /f "tokens=*" %%P in (
    'powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter 'Name=''chrome.exe''' | Where-Object { $_.CommandLine -like '*soccerdata*' -or $_.CommandLine -like '*--headless*' -or $_.CommandLine -like '*--host-resolver-rules*' -or $_.CommandLine -like '*--user-data-dir=*Temp\tmp*' } | Select-Object -ExpandProperty ProcessId)"'
) do (
    if not "%%P"=="" (
        echo [stop.bat] killing zombie chrome PID %%P
        taskkill /F /PID %%P >NUL 2>&1
    )
)

echo [stop.bat] Bot dihentikan. Start lagi dengan: .\run
