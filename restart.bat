@echo off
setlocal
cd /d "%~dp0"

REM ----------------------------------------------------------------
REM Restart the Hermes Football bot in one command: .\restart
REM (stop all bot.py instances + zombie Chrome, then run.bat)
REM ----------------------------------------------------------------
call "%~dp0stop.bat"
echo.
echo [restart.bat] Memulai ulang bot...
call "%~dp0run.bat" %*
