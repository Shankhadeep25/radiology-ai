@echo off
title Radiology AI Assistant - Local Runner
cd /d "%~dp0"

echo ===================================================
echo   Starting Radiology AI (Local Mode)
echo ===================================================
echo.

echo [1/2] Starting Backend API on http://localhost:8000 ...
start "Radiology AI - Backend" "%~dp0scripts\run_backend.bat"

echo [2/2] Starting Frontend on http://localhost:5173 ...
start "Radiology AI - Frontend" "%~dp0scripts\run_frontend.bat"

echo.
echo ===================================================
echo   Services are launching in separate windows:
echo   - Frontend: http://localhost:5173
echo   - Backend:  http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo.
echo   Demo Login:
echo     doctor@radiology.ai / demo1234
echo ===================================================
echo.

timeout /t 5 >nul 2>&1
start http://localhost:5173
