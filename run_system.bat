@echo off
title Anywear VTO - Live Virtual Try-On System
color 0A

echo ======================================================================
echo           ANYWEAR VTO - LIVE VIRTUAL TRY-ON SYSTEM LAUNCHER
echo ======================================================================
echo.

:: 1. Verify Python Installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: 2. Install / Verify Dependencies
echo [1/3] Checking dependencies...
python -m pip install -q -r "%~dp0server\requirements.txt"
if %errorlevel% neq 0 (
    echo [WARN] Pip install returned a non-zero exit code. Continuing to health check...
)

:: 3. Run Automated System Diagnostic
echo.
echo [2/3] Running automated system health check...
python "%~dp0server\health_check.py"
if %errorlevel% neq 0 (
    color 0E
    echo.
    echo [WARN] Diagnostic reported warnings. Press any key to start server anyway...
    pause >nul
)

:: 4. Launch Production Server
echo.
echo [3/3] Starting Anywear VTO WebSocket Streaming Server on port 8000...
echo ----------------------------------------------------------------------
echo WebSockets URL : ws://localhost:8000/ws/stream
echo Chrome Test URL: file:///%~dp0test_store.html
echo ----------------------------------------------------------------------
echo Press Ctrl+C in this window to stop the server.
echo.

cd /d "%~dp0server"
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server exited with error code %errorlevel%.
    pause
)
