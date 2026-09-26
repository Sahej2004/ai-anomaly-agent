@echo off
title AI Anomaly Detection Agent
cd /d "%~dp0"
echo ========================================================
echo   Launching Autonomous AI Anomaly Detection Agent...
echo ========================================================
echo.

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    py -3.13 -m streamlit run app.py
) else (
    python -m streamlit run app.py
)

pause
