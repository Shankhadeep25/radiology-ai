@echo off
title Radiology AI - Backend
cd /d "%~dp0..\backend"
call venv\Scripts\activate.bat
uvicorn app:app --reload --port 8000
if errorlevel 1 pause
