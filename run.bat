@echo off
setlocal

REM Work from this batch file's folder so the script is safe to run from
REM any directory (PowerShell: .\run or .\restart from anywhere).
cd /d "%~dp0"

REM ----------------------------------------------------------------
REM Pre-flight cleanup:
REM Kill zombie Chrome instances that were spawned by soccerdata's
REM Selenium/seleniumbase scrapers. We only target processes whose
REM command line mentions "soccerdata" OR "--headless" -- this is the
REM exact signature of the headless Chrome opened by the library.
REM Regular Chrome tabs you open manually for browsing are LEFT ALONE.
REM ----------------------------------------------------------------
for /f "tokens=*" %%P in (
    'powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter 'Name=''chrome.exe''' | Where-Object { $_.CommandLine -like '*soccerdata*' -or $_.CommandLine -like '*--headless*' -or $_.CommandLine -like '*--host-resolver-rules*' -or $_.CommandLine -like '*--user-data-dir=*Temp\tmp*' } | Select-Object -ExpandProperty ProcessId)"'
) do (
    if not "%%P"=="" (
        echo [run.bat] killing soccerdata zombie chrome PID %%P
        taskkill /F /PID %%P >NUL 2>&1
    )
)

REM Spawn the bot. The Python interpreter and any Chrome it later opens
REM will be isolated from your manual browsing sessions.
"%CD%\.venv\Scripts\python.exe" "%CD%\bot.py" %*